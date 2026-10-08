"""
Part 2 - Improved Authentic RAG : นำเข้าเอกสาร (ingest)

ใช้ LLM ตัวเดิม (qwen/qwen-2.5-72b-instruct ผ่าน OpenRouter) แต่ปรับปรุง 4 จุด
เทียบกับ part1_baseline/authenticRAG.py:

  [1] Chunking  : แบ่งตามหัวข้อ Markdown ก่อน แล้ว "ตัดซ้ำให้ขนาดไม่เกิน CHUNK_SIZE + มี overlap"
                  (baseline ใช้ MarkdownNodeParser อย่างเดียว -> บาง chunk ยาวทั้ง section)
  [2] Context   : ส่ง "เอกสารเต็ม" ให้ LLM เขียนบริบทของ chunk และให้ตอบเป็นภาษาไทย
                  (baseline ส่ง chunk ตัวเองแทนเอกสารเต็ม -> บริบทไม่ได้เพิ่มข้อมูลใหม่)
  [3] Embedding : ฝัง "บริบท + chunk" และเก็บบริบท/ชื่อไฟล์ไว้ใน vector index ด้วย
                  (baseline ไม่เก็บบริบทใน vector index -> ผลจาก dense search ไม่มีบริบทส่งให้ LLM)
  [4] BM25      : ใช้ analyzer "thai" ตัดคำภาษาไทย
                  (baseline ใช้ "standard" ซึ่งไม่ตัดคำไทย -> ทั้งประโยคกลายเป็น 1 token)

วิธีรัน (จากโฟลเดอร์หลักของโปรเจกต์):  python part2_improved/improvedRAG.py

ทางเลือก: ใช้ LLM บนเครื่องตัวเอง (Ollama) แทน OpenRouter (ฟรี ไม่ต้องใช้ credit) - ดู README
"""
import os
import re
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from openai import OpenAI
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

# ---------------- ค่าตั้งต้น ----------------
LLM_MODEL = "qwen/qwen-2.5-72b-instruct"   # LLM ตัวเดียวกับ baseline
EMBED_MODEL = "BAAI/bge-m3"                # embedding model ตัวเดียวกับ baseline
VECTOR_INDEX = "improved-vector-index"     # ใช้ index ใหม่ จะได้ไม่ทับของ baseline
BM25_INDEX = "improved-bm25-index"
CHUNK_SIZE = 512        # token ต่อ chunk (สูงสุด)
CHUNK_OVERLAP = 64      # token ที่ซ้อนกันระหว่าง chunk ติดกัน
MAX_DOC_CHARS = 40000   # กันเอกสารยาวเกิน context window ของ LLM
LLM_WORKERS = 4         # เรียก LLM พร้อมกันกี่ request (มากไปจะโดน rate limit)
LLM_RETRIES = 5         # ลองใหม่กี่ครั้งเมื่อเรียก LLM ไม่สำเร็จ
CHINESE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")  # Qwen บางครั้งหลุดไปตอบภาษาจีน

# LLM สำหรับสร้างบริบท: "openrouter" (ค่าเริ่มต้น) หรือ "ollama" (รันบนเครื่องตัวเอง)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-14b-16k")  # สร้างจาก Modelfile
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_NUM_CTX = 16384  # เอกสารยาวสุด ~13,400 token; ค่าเริ่มต้นของ Ollama (4096) จะตัดเอกสารทิ้งเงียบ ๆ
if LLM_PROVIDER == "ollama":
    LLM_WORKERS = 1     # GPU เครื่องเดียว ทำทีละ request
CONTEXT_CACHE = Path("results/contexts_cache.json")  # เก็บบริบทที่สร้างแล้ว รันซ้ำจะไม่เรียก LLM ซ้ำ

CONTEXT_PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Write the context in the same language as the document (Thai). Answer only with the succinct context and nothing else."""


class ImprovedContextualRAG:
    def __init__(self, opensearch_host="localhost", opensearch_port=9200):
        if LLM_PROVIDER == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            # timeout 60 วินาที: request ที่ค้างจะ error เร็ว แล้วให้ retry ของเราจัดการ
            self.llm = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=60, max_retries=0)
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        # แสดงว่าใช้โมเดลอะไร มาจากไหน
        if LLM_PROVIDER == "openrouter":
            print(f"LLM       : {LLM_MODEL}  ({self.llm.base_url})")
        else:
            print(f"LLM       : {OLLAMA_MODEL}  (Ollama {OLLAMA_URL}, context {OLLAMA_NUM_CTX} token)")
        print(f"Embedding : {EMBED_MODEL}  (HuggingFace, รันบน {self.embed_model.device})")
        self.os = OpenSearch(hosts=[{"host": opensearch_host, "port": opensearch_port}], use_ssl=False)

        # [1] Chunking 2 ขั้น: ตามหัวข้อ -> ตามขนาด
        self.md_parser = MarkdownNodeParser()
        self.splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # ---------- สร้าง index ----------
    def create_indices(self):
        """ลบ index เดิม (ถ้ามี) แล้วสร้างใหม่ เพื่อไม่ให้มี chunk เก่าค้างอยู่"""
        for name in (VECTOR_INDEX, BM25_INDEX):
            if self.os.indices.exists(index=name):
                self.os.indices.delete(index=name)

        dim = self.embed_model.get_sentence_embedding_dimension()
        self.os.indices.create(index=VECTOR_INDEX, body={
            "settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},  # เหมือน baseline
            "mappings": {"properties": {
                "embedding": {"type": "knn_vector", "dimension": dim},
                "content": {"type": "text"},
                "contextualized_content": {"type": "text"},   # [3] เก็บบริบทไว้ด้วย
                "source": {"type": "keyword"},
            }},
        })

        # [4] analyzer "thai" (มีในตัว OpenSearch) ตัดคำไทยเป็นคำ ๆ ก่อนทำ inverted index
        self.os.indices.create(index=BM25_INDEX, body={
            "settings": {"similarity": {"default": {"type": "BM25"}}},
            "mappings": {"properties": {
                "content": {"type": "text", "analyzer": "thai"},
                "contextualized_content": {"type": "text", "analyzer": "thai"},
                "source": {"type": "keyword"},
            }},
        })
        print(f"สร้าง index {VECTOR_INDEX} และ {BM25_INDEX} สำเร็จ")

    # ---------- [1] Chunking ----------
    def load_documents(self, md_paths):
        """คืนค่า list ของ (ชื่อไฟล์, เนื้อหาเอกสารเต็ม, [chunk, ...])"""
        docs = []
        for path in md_paths:
            text = Path(path).read_text(encoding="utf-8")
            sections = self.md_parser.get_nodes_from_documents([Document(text=text)])
            chunks = self.splitter.get_nodes_from_documents(sections)
            docs.append((Path(path).name, text, [c.text for c in chunks]))
            print(f"{Path(path).name}: {len(sections)} sections -> {len(chunks)} chunks")
        return docs

    # ---------- [2] Contextual Retrieval ----------
    def chat(self, prompt, temperature=0.1):
        """ส่ง prompt ให้ LLM แล้วคืนข้อความตอบ (OpenRouter หรือ Ollama)"""
        if LLM_PROVIDER == "openrouter":
            res = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=temperature,
            )
            return res.choices[0].message.content.strip()

        # Ollama native API (ต้องใช้ตัวนี้เพราะตั้ง num_ctx ได้)
        body = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 200, "temperature": temperature},
        }
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as res:
            return json.load(res)["message"]["content"].strip()

    def generate_context(self, full_doc, chunk):
        prompt = CONTEXT_PROMPT.format(doc=full_doc[:MAX_DOC_CHARS], chunk=chunk)
        text = ""
        for attempt in range(LLM_RETRIES):
            try:
                text = self.chat(prompt, temperature=0.1 if attempt == 0 else 0.7)  # ลองใหม่ให้สุ่มมากขึ้น
                if not CHINESE.search(text):
                    return text
                tqdm.write(f"บริบทมีภาษาจีน ขอให้เขียนใหม่ (attempt {attempt + 1}/{LLM_RETRIES})")
                continue
            except Exception as e:
                # ส่วนใหญ่คือโดน rate limit (429) หรือ timeout -> รอแล้วลองใหม่ (5, 10, 20, 40 วินาที)
                tqdm.write(f"Error calling LLM (attempt {attempt + 1}/{LLM_RETRIES}): {e}")
                if attempt < LLM_RETRIES - 1:
                    time.sleep(5 * 2 ** attempt)
        return text   # ลองครบแล้วยังไม่ได้: คืนข้อความสุดท้าย (หรือ "" ถ้า error) - chunk ยังค้นด้วยเนื้อหาได้

    # ---------- นำเข้า OpenSearch ----------
    def add_documents(self, docs):
        items = [(source, full_doc, chunk) for source, full_doc, chunks in docs for chunk in chunks]

        # โหลดบริบทที่เคยสร้างสำเร็จแล้ว (key = เนื้อหา chunk) เรียก LLM เฉพาะ chunk ที่ยังไม่มี
        cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8")) if CONTEXT_CACHE.exists() else {}
        # chunk ที่ยังไม่มีบริบท หรือบริบทเป็นภาษาจีน -> สร้างใหม่
        todo = [it for it in items if not cache.get(it[2]) or CHINESE.search(cache[it[2]])]
        print(f"\nกำลังสร้างบริบท: {len(todo)} chunks (มีในแคชแล้ว {len(items) - len(todo)})")

        # เรียก LLM แบบขนาน แสดง progress bar และบันทึกแคชทุกครั้งที่ได้บริบทใหม่
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            results = pool.map(lambda it: self.generate_context(it[1], it[2]), todo)
            for (_, _, chunk), ctx in tqdm(zip(todo, results), total=len(todo)):
                if ctx:
                    cache[chunk] = ctx
                    CONTEXT_CACHE.parent.mkdir(exist_ok=True)
                    CONTEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        contexts = [cache.get(chunk, "") for _, _, chunk in items]

        # [3] ฝัง "บริบท + chunk" เป็น vector ทีเดียวทั้ง batch
        texts = [f"{ctx}\n\n{chunk}" for (_, _, chunk), ctx in zip(items, contexts)]
        embeddings = self.embed_model.encode(texts, batch_size=16, show_progress_bar=True)

        actions = []
        for i, ((source, _, chunk), ctx, emb) in enumerate(zip(items, contexts, embeddings)):
            doc_id = f"{source}#{i}"
            fields = {"content": chunk, "contextualized_content": ctx, "source": source}
            actions.append({"_index": VECTOR_INDEX, "_id": doc_id, **fields, "embedding": emb.tolist()})
            actions.append({"_index": BM25_INDEX, "_id": doc_id, **fields})

        helpers.bulk(self.os, actions)
        self.os.indices.refresh(index=f"{VECTOR_INDEX},{BM25_INDEX}")
        print(f"\nนำเข้า {len(items)} chunks สำเร็จ")
        failed = sum(ctx == "" or bool(CHINESE.search(ctx)) for ctx in contexts)
        if failed:
            print(f"คำเตือน: สร้างบริบทไม่สำเร็จ {failed}/{len(items)} chunks (ดู error ด้านบน) - รันใหม่เพื่อเติมเฉพาะที่ขาด")


def main():
    md_paths = sorted(str(p) for p in Path("./corpus_input").glob("*.md"))
    rag = ImprovedContextualRAG()
    rag.create_indices()
    docs = rag.load_documents(md_paths)
    rag.add_documents(docs)


if __name__ == "__main__":
    main()
