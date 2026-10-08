"""
Part 2 - Improved Authentic RAG : ค้นเอกสาร วิเคราะห์ และสรุปคำตอบ

สืบทอดจาก AuthenticSearchRAG ของ baseline เพื่อให้เห็นชัดว่า "เปลี่ยนแค่ตรงไหน"
  - ใช้ index ใหม่ที่สร้างจาก improvedRAG.py (ตัดคำไทย + บริบทจากเอกสารเต็ม + chunk ขนาดคงที่)
  - [5] Reranking: ดึง candidates จาก hybrid search (BM25 + vector + RRF) มา 20 อัน
        แล้วให้ cross-encoder (BAAI/bge-reranker-v2-m3) ให้คะแนนใหม่ เหลือ k อันที่ดีที่สุด
  - การสร้างคำตอบสุดท้าย: LLM ตัวเดิม prompt เดิม (ไม่เปลี่ยน)

วิธีรัน (จากโฟลเดอร์หลักของโปรเจกต์):  python part2_improved/onlysearchImprovedRAG.py
"""
import os
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part1_baseline"))
from onlysearchAuthenticRAG import AuthenticSearchRAG  # noqa: E402

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder หลายภาษา รองรับภาษาไทย
NUM_CANDIDATES = 20                        # จำนวน candidates ก่อน rerank


class ImprovedSearchRAG(AuthenticSearchRAG):
    def __init__(self, opensearch_host="localhost", opensearch_port=9200):
        super().__init__(opensearch_host, opensearch_port)
        self.vector_index_name = "improved-vector-index"
        self.bm25_index_name = "improved-bm25-index"
        self.reranker = CrossEncoder(RERANK_MODEL)
        print(f"Reranker  : {RERANK_MODEL}  (HuggingFace, รันบน {self.reranker.device})")

    def hybrid_search(self, query, k=5, rrf_k=60):
        """hybrid search เดิม (BM25 + vector + RRF) แล้วตามด้วย reranking"""
        candidates = super().hybrid_search(query, k=NUM_CANDIDATES, rrf_k=rrf_k)

        # cross-encoder อ่าน "คำถาม + เอกสาร" พร้อมกัน จึงให้คะแนนความเกี่ยวข้องได้แม่นกว่า
        pairs = [(query, f"{src.get('contextualized_content', '')}\n\n{src.get('content', '')}")
                 for _, _, src in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        print(f"Reranked {len(candidates)} candidates -> top {k}")

        return [(doc_id, float(score), src) for (doc_id, _, src), score in reranked[:k]]


def main():
    if "OPENROUTER_API_KEY" not in os.environ and os.environ.get("LLM_PROVIDER") != "ollama":
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        return

    rag = ImprovedSearchRAG(opensearch_host="localhost", opensearch_port=9200)

    # ชุดคำถามเดียวกับ baseline เพื่อเปรียบเทียบคำตอบ
    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]
    results = rag.search_multiple_questions(questions, k=5)
    rag.export_results_to_json(results, "improved_rag_search_results.json")


if __name__ == "__main__":
    main()
