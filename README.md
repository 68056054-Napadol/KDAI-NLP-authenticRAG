# KDAI-NLP-authenticRAG

Assignment: Authentic RAG (Contextual Retrieval) — ตามสไลด์ PART4 หน้า 22 (Hands-on 1) และหน้า 25 (Hands-on 2)

| ข้อ | ทำอะไร | ไฟล์ |
|---|---|---|
| (1) | สร้างระบบ RAG ตามสไลด์ | [part1_baseline/](part1_baseline/) |
| (2) | ปรับปรุง chunking, embedding, search, reranking (ไม่เปลี่ยน LLM) | [part2_improved/](part2_improved/) |
| (3) | อธิบายความต่าง + วัดผล | [ข้อ (3)](#3-ความแตกต่างระหว่าง-1-และ-2) + [compare_retrieval.py](compare_retrieval.py) |
| | อธิบายละเอียด: ทำงานยังไง, โค้ดตรงไหน, ผลแต่ละจุด | [EXPLANATION.md](EXPLANATION.md) |

```
KDAI-NLP-authenticRAG/
├── corpus_input/                      # เอกสาร 4 ไฟล์ (หัดเยอรมัน, อหิวาตกโรค, ต้อกระจก, กรดไหลย้อน)
├── part1_baseline/
│   ├── authenticRAG.py                # (1) นำเข้าเอกสาร  -> anthropic-*-index
│   └── onlysearchAuthenticRAG.py      # (1) ค้น + ตอบ
├── part2_improved/
│   ├── improvedRAG.py                 # (2) นำเข้าเอกสารแบบปรับปรุง -> improved-*-index
│   └── onlysearchImprovedRAG.py       # (2) ค้น + rerank + ตอบ
├── compare_retrieval.py               # (3) วัด Hit@5 / MRR@5 ของทั้ง 2 ระบบ
├── EXPLANATION.md                     # (3) อธิบายละเอียด baseline vs improved พร้อมอ้างอิงโค้ด
├── index_backup.py                    # สำรอง/กู้คืน index ไปเครื่องอื่น
├── Modelfile                          # สร้างโมเดล Ollama สำหรับรันบนเครื่อง (ไม่บังคับ)
└── results/                           # ผลวัด, แคชบริบท, ไฟล์สำรอง index
```

---

## วิธีรัน

ต้องมี Docker (OpenSearch), Python 3.10+, และ API key ของ [OpenRouter](https://openrouter.ai/keys)

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.19.1
```

```bash
pip install -r requirements.txt
```

ตั้งค่า API key และให้ Python แสดงภาษาไทยได้ (PowerShell, ตั้งครั้งเดียวต่อ terminal):

```bash
$env:OPENROUTER_API_KEY="your_api_key_here"
```

```bash
$env:PYTHONUTF8="1"
```

รันทุกคำสั่งจากโฟลเดอร์หลักของโปรเจกต์:

```bash
python part1_baseline/authenticRAG.py
```

```bash
python part1_baseline/onlysearchAuthenticRAG.py
```

```bash
python part2_improved/improvedRAG.py
```

```bash
python part2_improved/onlysearchImprovedRAG.py
```

```bash
python compare_retrieval.py
```

ทั้ง 2 ระบบใช้ index แยกกัน (`anthropic-*` vs `improved-*`) จึงเก็บไว้เทียบกันได้

บริบทที่สร้างแล้วถูกเก็บใน `results/contexts_cache.json` — รัน `improvedRAG.py` ซ้ำจะเรียก LLM เฉพาะ chunk ที่ยังไม่มีบริบท

### แต่ละคำสั่งได้อะไรออกมา

| คำสั่ง | ผลลัพธ์ | เก็บที่ไหน |
|---|---|---|
| `python part1_baseline/authenticRAG.py` | index `anthropic-vector-index`, `anthropic-bm25-index` (53 chunk) | ใน OpenSearch (Docker) — ไม่มีไฟล์ในโปรเจกต์ |
| `python part2_improved/improvedRAG.py` | index `improved-vector-index`, `improved-bm25-index` (212 chunk) | ใน OpenSearch (Docker) |
| | `results/contexts_cache.json` | ไฟล์ในโปรเจกต์ |
| `python part1_baseline/onlysearchAuthenticRAG.py` | `authentic_rag_search_results.json` | ไฟล์ในโปรเจกต์ |
| `python part2_improved/onlysearchImprovedRAG.py` | `improved_rag_search_results.json` | ไฟล์ในโปรเจกต์ |
| `python compare_retrieval.py` | `retrieval_comparison.json` | ไฟล์ในโปรเจกต์ |
| `python index_backup.py export` | `results/indices/*.json` (4 ไฟล์) | ไฟล์ในโปรเจกต์ |
| `python index_backup.py import` | index ทั้ง 4 อัน (อ่านจาก `results/indices/`) | ใน OpenSearch (Docker) |

ทุกคำสั่ง**เขียนทับ**ไฟล์/index เดิม ไม่ต้องลบอะไรก่อนรัน

**แต่ละไฟล์คืออะไร**

| ไฟล์ | เนื้อหา | ใช้ทำอะไร |
|---|---|---|
| `authentic_rag_search_results.json` | 5 คำถามจากสไลด์ + คำตอบของ baseline + 5 chunk ที่ค้นได้ (เนื้อหา, บริบท, คะแนน) | ดูคำตอบสุดท้ายของระบบ (1) |
| `improved_rag_search_results.json` | เหมือนข้างบน แต่เป็นของระบบ improved (คะแนน = คะแนนจาก reranker) | ดูคำตอบสุดท้ายของระบบ (2) และเทียบกับ (1) |
| `retrieval_comparison.json` | Hit@5, MRR@5, Hit@1, ความยาวข้อความที่ส่งให้ LLM ของทั้ง 2 ระบบ + อันดับของ chunk ที่ถูกในแต่ละคำถาม (14 ข้อ) | ตัวเลขสำหรับตอบข้อ (3) |
| `results/contexts_cache.json` | บริบทที่ LLM เขียนให้แต่ละ chunk ของ improved (key = เนื้อหา chunk) | รันซ้ำไม่ต้องเรียก LLM ใหม่ (ประหยัด credit/เวลา) — **ห้ามลบ** ถ้าไม่ได้ตั้งใจสร้างบริบทใหม่ทั้งหมด |
| `results/indices/*.json` | สำเนา index ทั้ง 4 อัน (settings, mapping, เอกสาร + vector) | ย้ายไปเครื่องอื่นโดยไม่ต้อง ingest ใหม่ |
| `results/retrieval_no_llm_context.json` | ผลวัดรอบแรกที่ทั้ง 2 ระบบยังไม่มีบริบทจาก LLM | ผล ablation ที่อ้างถึงในข้อ (3) |
| `improved_ingest.log` | ข้อความทั้งหมดที่แสดงบนจอระหว่าง ingest (มีเฉพาะเมื่อรันพร้อม `\| Tee-Object improved_ingest.log`) | ไล่หา error ทีหลัง |

นอกโปรเจกต์: โมเดล `bge-m3` และ `bge-reranker-v2-m3` ถูกดาวน์โหลดไว้ที่ `C:\Users\<ชื่อผู้ใช้>\.cache\huggingface\` (~4.5 GB) ครั้งแรกครั้งเดียว

### ย้ายไปเครื่องอื่นโดยไม่ต้อง ingest ใหม่

index ทั้ง 4 อันอยู่ใน Docker ของ OpenSearch (ไม่ได้ไปกับ git) จึงสำรองเป็นไฟล์ไว้ที่ `results/indices/` (~7 MB)

เครื่องที่ ingest แล้ว (ทำหลัง ingest เสร็จทุกครั้ง แล้ว commit ไฟล์):

```bash
python index_backup.py export
```

เครื่องใหม่ (หลังรัน OpenSearch + `pip install`) — ข้ามขั้น ingest ไปรันค้นหา/ตอบได้เลย:

```bash
python index_backup.py import
```

### ทางเลือก: รันทั้งหมดบนเครื่อง ไม่ต้องใช้ OpenRouter (Ollama, ฟรี)

ตั้ง `LLM_PROVIDER=ollama` แล้วทุกสคริปต์ (สร้างบริบท + ตอบคำถาม ทั้ง 2 ระบบ) จะเรียก LLM ผ่าน [Ollama](https://ollama.com/download) แทน
ถ้าไม่ตั้ง โค้ดทำงานเหมือนในสไลด์ทุกอย่าง (Qwen 2.5 72B ผ่าน OpenRouter)
ทั้ง 2 ระบบยังใช้ LLM ตัวเดียวกัน จึงเปรียบเทียบกันได้เหมือนเดิม

#### โมเดลบนเครื่อง vs โมเดลในสไลด์ ต่างกันอย่างไร

| | `qwen2.5:14b` (Ollama, บนเครื่อง) | `qwen/qwen-2.5-72b-instruct` (OpenRouter, ในสไลด์) |
|---|---|---|
| ตระกูล | Qwen 2.5 Instruct (ตระกูลเดียวกัน, tokenizer เดียวกัน) | Qwen 2.5 Instruct |
| ขนาด (parameter) | 14.7 พันล้าน | 72.7 พันล้าน (~5 เท่า) |
| ความละเอียดตัวเลข | ย่อเหลือ 4-bit (quantized) ให้ใส่การ์ดจอบ้านได้ | 8–16 bit (แล้วแต่ผู้ให้บริการ) |
| ขนาดไฟล์ | ~9 GB | ~145 GB |
| รันที่ไหน | GPU ของเรา | GPU ของผู้ให้บริการ (data center) |
| ค่าใช้จ่าย | ฟรี | จ่ายตาม token |
| คุณภาพ | พอใช้ (เขียนบริบทสั้น ๆ ได้ดี) | ดีกว่า: ภาษาไทยเป็นธรรมชาติ ทำตามคำสั่งแม่น อ่านเอกสารยาวได้ดี |

- **parameter** = ตัวเลขที่โมเดลเรียนรู้มา (B = billion/พันล้าน) ยิ่งมากยิ่งฉลาด แต่ยิ่งใช้หน่วยความจำมาก
- **Instruct** = รุ่นที่ฝึกต่อให้ "ทำตามคำสั่ง/ตอบคำถาม" (รุ่น base แค่ต่อข้อความไปเรื่อย ๆ) — `qwen2.5:14b` ของ Ollama เป็นรุ่น Instruct โดยค่าเริ่มต้น
- **ผลกระทบต่อการบ้าน:** ตัวเลขที่ได้จากโมเดลต่างกันเอามาเทียบกันตรง ๆ ไม่ได้ แต่การเทียบ (1) vs (2) ยังยุติธรรม ตราบใดที่**ทั้ง 2 ระบบใช้ LLM ตัวเดียวกัน** — เวลารายงานผลให้ระบุว่าใช้ LLM ตัวไหน

#### ติดตั้งครั้งเดียว

```bash
ollama pull qwen2.5:14b
```

```bash
ollama create qwen2.5-14b-16k -f Modelfile
```

**ทำไมต้อง `ollama create`:** Ollama ให้โมเดลอ่านข้อความได้ครั้งละ **4,096 token โดยค่าเริ่มต้น** (context window)
แต่ขั้นสร้างบริบทต้องส่ง "เอกสารเต็ม" ไปด้วย ซึ่งยาวถึง ~13,400 token (ไฟล์ `1.md`) — ถ้าเกิน Ollama จะ**ตัดส่วนเกินทิ้งเงียบ ๆ ไม่มี error** โมเดลจึงเห็นเอกสารไม่ครบ และบริบทที่ได้ก็ผิด

| ไฟล์ | ตัวอักษร | token (Qwen) |
|---|---|---|
| 1.md | 23,967 | 13,414 |
| 2.md | 22,537 | 12,745 |
| 44.md | 18,077 | 9,757 |
| 5555.md | 22,091 | 11,873 |

[Modelfile](Modelfile) สร้างโมเดลชื่อใหม่ `qwen2.5-14b-16k` = `qwen2.5:14b` เดิม + `PARAMETER num_ctx 16384` (อ่านได้ 16k token)
ต้องทำแบบนี้เพราะ baseline เรียก Ollama ผ่าน API แบบ OpenAI ซึ่ง**ส่งค่า context window ไปกับ request ไม่ได้** จึงต้องฝังค่าไว้ในตัวโมเดลเลย
(ไม่ได้ดาวน์โหลดอะไรเพิ่ม — แค่สร้าง "ชื่อเล่น" ที่ชี้ไปยังไฟล์โมเดลเดิมพร้อมค่าตั้งใหม่)

context ที่ยาวขึ้นใช้ VRAM เพิ่ม (~3 GB สำหรับ 16k ใน 14B) จึงไม่ตั้งไว้ใหญ่เกินจำเป็น

**ทำไม OpenRouter ไม่ต้องทำ Modelfile:** context window เป็นค่าที่ฝั่ง "เซิร์ฟเวอร์ที่รันโมเดล" ตั้ง ไม่ใช่ตัวโมเดล (Qwen 2.5 รองรับ 32k ทั้ง 14B และ 72B)
ผู้ให้บริการบน OpenRouter ใช้ GPU ระดับ data center จึงเปิดเต็ม 32k ไว้แล้ว และถ้า prompt ยาวเกินจะตอบ error กลับมา —
ส่วน Ollama ออกแบบให้รันบนเครื่องบ้าน จึงตั้งค่าเริ่มต้นไว้ต่ำ (4,096) เพื่อประหยัด VRAM และตัดส่วนเกินทิ้งโดยไม่แจ้ง

ก่อนรันสคริปต์ทุกครั้ง (ใน terminal เดียวกัน):

```bash
$env:LLM_PROVIDER="ollama"
```

ถ้าเคยสร้างบริบทด้วย OpenRouter ไว้ ให้ลบ `results/contexts_cache.json` ก่อน เพื่อให้บริบททั้งหมดมาจากโมเดลเดียวกัน

**สเปกเครื่องขั้นต่ำ** (embedding `bge-m3` + reranker `bge-reranker-v2-m3` จะโหลดจาก HuggingFace อัตโนมัติ ~4.5 GB และรันบนเครื่องอยู่แล้ว)

| | ขั้นต่ำ | แนะนำ |
|---|---|---|
| GPU (NVIDIA) | 8 GB VRAM + `qwen2.5:7b` | 16 GB VRAM + `qwen2.5:14b` |
| RAM | 16 GB | 32 GB |
| พื้นที่ดิสก์ว่าง | ~20 GB | ~25 GB |

- ใช้ `qwen2.5:7b` ให้แก้บรรทัด `FROM` ใน Modelfile เป็น `qwen2.5:7b` แล้วสร้างใหม่
- ไม่มี GPU ก็รันได้ (RAM ≥ 32 GB) แต่ช้ามาก — หลายนาทีต่อ chunk รวมหลายชั่วโมง
- โมเดลเล็กตอบภาษาไทยได้ด้อยกว่า Qwen 72B คำตอบสุดท้ายจึงอาจสั้น/ผิดมากกว่า แต่ไม่กระทบการเทียบ (1) vs (2)

---

## (1) ระบบ RAG ตามสไลด์ (baseline)

โค้ดจาก Hands-on 1 และ 2 ([aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)) ใช้เวอร์ชันแก้สำหรับ Windows (`TextLoader(..., encoding="utf-8")`)

```
นำเข้า:  Markdown -> MarkdownNodeParser (แบ่งตามหัวข้อ)
          -> LLM เขียน "บริบท" ให้แต่ละ chunk (Contextual Retrieval)
          -> bge-m3 embedding  -> anthropic-vector-index (kNN, cosine)
          -> BM25 (standard)   -> anthropic-bm25-index
ค้น:     คำถาม -> BM25 top 10 + Vector top 10 -> RRF -> top 5 -> Qwen 2.5 72B ตอบ
```

## (2) ระบบที่ปรับปรุง (improved)

LLM ตัวเดียวกับ baseline เสมอ (ค่าเริ่มต้น `qwen/qwen-2.5-72b-instruct` ผ่าน OpenRouter หรือ Qwen 14B ผ่าน Ollama เมื่อตั้ง `LLM_PROVIDER=ollama`) ทั้งตอนเขียนบริบทและตอนตอบ, prompt ตอบคำถามเดิม, embedding model เดิม (`BAAI/bge-m3`) — เปลี่ยนแค่ 5 จุดนี้

| # | ส่วน | baseline (1) | improved (2) |
|---|---|---|---|
| 1 | Chunking | `MarkdownNodeParser` อย่างเดียว | แบ่งตามหัวข้อ แล้วตัดให้ไม่เกิน 512 token, overlap 64 |
| 2 | Context | ส่ง **chunk ตัวเอง** แทนเอกสารเต็ม | ส่ง **เอกสารเต็ม** + ขอบริบทเป็นภาษาไทย |
| 3 | Embedding / vector index | เก็บแค่ content | ฝัง "บริบท + chunk" และเก็บบริบทไว้ใน vector index ด้วย |
| 4 | BM25 analyzer | `standard` | `thai` (ตัดคำไทย) |
| 5 | Reranking | ไม่มี (RRF top 5) | RRF top 20 → cross-encoder `BAAI/bge-reranker-v2-m3` → top 5 |

ไฟล์ค้นหา [onlysearchImprovedRAG.py](part2_improved/onlysearchImprovedRAG.py) **สืบทอด** (subclass) จาก baseline จึงเห็นได้ชัดว่าเปลี่ยนแค่ชื่อ index กับเพิ่ม reranking

---

## (3) ความแตกต่างระหว่าง (1) และ (2)

### สรุปสั้น ๆ

| | (1) baseline | (2) improved | ดีขึ้นเพราะ |
|---|---|---|---|
| Hit@5 (เจอ chunk ที่ถูกใน 5 อันดับแรก) | 1.00 | 1.00 | – |
| **MRR@5** (ยิ่งสูง chunk ที่ถูกยิ่งอยู่อันดับต้น) | 0.81 | **0.96** | บริบทจากเอกสารเต็ม + reranking |
| **Hit@1** (อันดับ 1 ถูกเลย) | 0.71 (10/14) | **0.93** (13/14) | บริบทจากเอกสารเต็ม + reranking |
| ข้อความที่ส่งให้ LLM ต่อคำถาม | 10,257 ตัวอักษร | **2,408** ตัวอักษร (น้อยลง 4.3 เท่า) | chunk เล็ก + เลือกแม่นขึ้น |
| BM25 คืนผลลัพธ์ได้ (จาก 14 คำถาม) | **2** คำถาม | **14** คำถาม | ตัดคำไทยได้ |
| จำนวน chunk | 53 (ยาว 112–8,404 ตัวอักษร) | 212 (ยาว 112–728 ตัวอักษร) | chunk ขนาดสม่ำเสมอ |

อันดับของ chunk ที่ถูกในแต่ละคำถาม (`-` = ไม่อยู่ใน 5 อันดับแรก):

```
baseline : 1  1  4  1  1  3  1  1  1  1  5  1  1  2
improved : 1  1  1  1  1  1  1  1  1  1  2  1  1  1
```

> วัดด้วย [compare_retrieval.py](compare_retrieval.py) (14 คำถามที่รู้คำตอบ ครอบคลุมทั้ง 4 เอกสาร) ผลดิบ: [retrieval_comparison.json](retrieval_comparison.json)
> **LLM ที่ใช้รอบนี้: Qwen 2.5 14B ผ่าน Ollama (`qwen2.5-14b-16k`) ทั้ง 2 ระบบ** ทั้งขั้นสร้างบริบทและขั้นตอบ
> (เดิมตั้งใจใช้ Qwen 2.5 72B ผ่าน OpenRouter ตามสไลด์ แต่ credit หมดระหว่างสร้างบริบท — เปลี่ยนทั้ง 2 ระบบพร้อมกัน การเทียบจึงยังยุติธรรม)

### อธิบายทีละจุด: ต่างกันตรงไหน ดีขึ้นอย่างไร เพราะอะไร

**[4] BM25: `standard` → `thai` analyzer (สำคัญที่สุด)**

- **ต่างกัน:** baseline ตัด token ด้วย `standard` analyzer ซึ่ง**ไม่รู้จักการตัดคำไทย** (ภาษาไทยไม่มีช่องว่างระหว่างคำ) จึงได้ทั้งประโยคเป็น 1 token
  ```
  "โรคหัดเยอรมันมีอันตรายกับหญิงตั้งครรภ์"
  standard -> [โรคหัดเยอรมันมีอันตรายกับหญิงตั้งครรภ์]                (1 token)
  thai     -> [โรค] [หัดเยอรมัน] [อันตราย] [หญิง] [ครรภ์]           (5 tokens)
  ```
- **ดีขึ้น:** BM25 ของ baseline คืนผลได้แค่ 2 จาก 14 คำถาม และเจอ chunk ที่ถูกแค่ 1 คำถาม (Hit@5 = 0.07) → ของ improved เจอ chunk ที่ถูกทุกคำถาม (Hit@5 = 1.00)
- **เพราะ:** BM25 ให้คะแนนจาก TF/DF ของ term ใน inverted index (สไลด์หน้า 24) ถ้า term ใน index เป็น "ทั้งประโยค" ก็แทบไม่มีทางตรงกับคำถาม
  ผลคือ "hybrid search" ของ baseline จริง ๆ แล้วเป็น **vector search อย่างเดียว** (ส่วน BM25 ใช้ไม่ได้)

**[1] Chunking: แบ่งตามหัวข้ออย่างเดียว → แบ่งตามหัวข้อ + จำกัดขนาด 512 token (overlap 64)**

- **ต่างกัน:** baseline เขียน `MarkdownNodeParser(chunk_size=256)` แต่ `chunk_size` **ไม่ใช่พารามิเตอร์ของ parser นี้** (ถูกเมิน) จึงได้ chunk = ทั้ง section บางอันยาวถึง 8,404 ตัวอักษร
- **ดีขึ้น:** chunk ยาวสม่ำเสมอ (112–728 ตัวอักษร) และส่งข้อความให้ LLM น้อยลง 4.3 เท่าต่อคำถาม
- **เพราะ:** chunk ยาวมาก ๆ ถูกบีบเป็น vector เดียว ความหมายจึง "เจือจาง" และส่งเนื้อหาที่ไม่เกี่ยวให้ LLM เยอะ (เปลือง token, LLM หาคำตอบยากขึ้น)
  chunk เล็กแม่นกว่า แต่เสี่ยง "ขาดบริบท" → แก้ด้วยจุด [2]

**[2] Contextual Retrieval: ส่ง chunk ตัวเอง → ส่งเอกสารเต็ม**

- **ต่างกัน:** baseline มีบั๊ก `full_doc_content = doc.page_content` (ใน `add_documents_with_context`) คือส่ง **chunk ตัวเอง** เป็น "เอกสาร" ให้ LLM บริบทที่ได้จึงแค่สรุป chunk ซ้ำ ไม่มีข้อมูลใหม่
  improved ส่ง **เอกสารเต็ม** ตามแนวทาง Anthropic และขอให้เขียนบริบทเป็นภาษาไทย (BM25 ภาษาไทยจะได้ใช้ประโยชน์จากบริบทด้วย)
- **ตัวอย่างจริง:** chunk `2.md#96` ขึ้นต้นว่า `(Ringer lactate) หรืออะซีทาร์ (Acetar) แต่ถ้าไม่มีอาจใช้น้ำเกลือนอร์มัล ...` — ตัว chunk ไม่ได้บอกเลยว่าพูดถึงโรคอะไร
  บริบทที่ LLM เขียนจากเอกสารเต็ม: *"ในส่วนของการรักษาผู้ป่วยที่มีอาการรุนแรงของอหิวาตกโรค โดยแพทย์จะให้สารน้ำและเกลือแร่ทางหลอดเลือดดำ ..."* → คำถามเรื่องอหิวาต์จึงค้นเจอ chunk นี้
- **ดีขึ้น (วัดได้):** vector search ของ improved ได้ MRR **0.75 → 0.90** เมื่อเพิ่มบริบท ส่วนของ baseline ได้ **0.86 → 0.86 (ไม่ขยับเลย)** — ยืนยันว่าบริบทของ baseline ไม่ได้เพิ่มข้อมูลใหม่เพราะบั๊กข้างบน
- **เพราะ:** chunk เล็กเสี่ยง "ขาดบริบท" (ไม่รู้ว่าเป็นของโรคไหน หัวข้อไหน) ซึ่งเป็นสิ่งที่ Contextual Retrieval ถูกออกแบบมาแก้ (สไลด์หน้า 18–19: Anthropic รายงานว่าลดการค้นพลาดได้ 35% ด้วย contextual embedding, 49% เมื่อรวม contextual BM25 และ 67% เมื่อเพิ่ม reranking)

**[3] Embedding / vector index เก็บบริบทด้วย**

- **ต่างกัน:** baseline เก็บ `contextualized_content` แค่ใน BM25 index ส่วน vector index ไม่มี; และ RRF เก็บ `_source` จาก ranker ที่เจอเอกสารก่อน
- **ดีขึ้น/เพราะ:** chunk ที่เจอจาก vector search อย่างเดียวจะถูกส่งให้ LLM **โดยไม่มีบริบท** ใน baseline — improved เก็บบริบท (และชื่อไฟล์) ไว้ทั้ง 2 index จึงส่งบริบทให้ LLM ได้ทุก chunk

**[5] Reranking ด้วย cross-encoder**

- **ต่างกัน:** baseline เอา top 5 จาก RRF ไปใช้เลย; improved ดึง 20 อันดับแรก แล้วให้ `BAAI/bge-reranker-v2-m3` ให้คะแนนใหม่ เหลือ 5
- **ดีขึ้น:** MRR@5 ของ improved hybrid **0.89 → 0.96** หลัง rerank
- **เพราะ:** embedding (bi-encoder) แปลงคำถามกับเอกสาร**แยกกัน**แล้ววัดระยะ → เร็วแต่หยาบ;
  cross-encoder อ่าน**คำถาม + เอกสารพร้อมกัน** → เห็นความสัมพันธ์ระดับคำ จึงแม่นกว่า แต่ช้า เลยใช้กับแค่ 20 candidates
  นอกจากนี้ RRF ใช้แค่ "อันดับ" ไม่สนคะแนนจริง ถ้า ranker ตัวหนึ่งผลแย่ (อย่าง BM25 ของ baseline) ก็ดึงอันดับรวมลง — reranker ช่วยจัดลำดับใหม่ด้วยเนื้อหาจริง

### Ablation (แยกผลทีละส่วน)

MRR@5 ของแต่ละวิธีค้น — วัด 2 รอบ: ก่อนมีบริบทจาก LLM ([results/retrieval_no_llm_context.json](results/retrieval_no_llm_context.json)) และหลังมีบริบทครบ

| วิธีค้น | baseline ไม่มีบริบท | baseline มีบริบท | improved ไม่มีบริบท | improved มีบริบท |
|---|---|---|---|---|
| BM25 อย่างเดียว | 0.07 | 0.07 | 0.71 | 0.74 |
| Vector อย่างเดียว | 0.86 | 0.86 | 0.75 | **0.90** |
| Hybrid (RRF) | 0.82 | 0.81 | 0.77 | 0.89 |
| Hybrid + rerank | – | – | 0.87 | **0.96** |

อ่านตารางนี้ได้ว่า
1. **BM25 ของ baseline ใช้ไม่ได้กับภาษาไทย** (0.07) แม้จะมีบริบทภาษาไทยแล้วก็ตาม เพราะปัญหาอยู่ที่ analyzer และเมื่อนำไปรวมด้วย RRF ยัง**ดึงผลลง** (0.86 → 0.81)
2. **บริบทของ baseline ไม่ช่วยเลย** (vector 0.86 → 0.86) เพราะ LLM ไม่เคยเห็นเอกสารเต็ม
3. **chunk เล็กอย่างเดียวทำให้ vector search แย่ลง** (0.86 → 0.75) แต่เมื่อเพิ่มบริบทจากเอกสารเต็ม กลับ**ดีกว่า baseline** (0.90) — chunking [1] กับ contextual retrieval [2] ต้องใช้คู่กัน
4. **reranking ยกผลขึ้นอีกขั้น** (0.89 → 0.96)

### ตัวอย่างคำตอบสุดท้าย

ดูคำตอบเต็มทั้ง 5 ข้อได้ที่ [authentic_rag_search_results.json](authentic_rag_search_results.json) และ [improved_rag_search_results.json](improved_rag_search_results.json)

**"ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?"**

- **baseline:** *"…มีโอกาสติดเชื้อน้อยมากหรือแทบไม่มีเลย … ป้องกันได้เกือบ 100%…"* — ความรู้ทั่วไป ไม่ได้มาจากเอกสาร
- **improved:** *"…มักจะมีภูมิคุ้มกันตลอดชีวิต … แต่ยังมีความเป็นไปได้ว่าการได้รับเชื้อใหม่อาจทำให้ติดเชื้อได้ แต่จะไม่แสดงอาการและเชื้อจะอยู่แค่ที่ทางเดินหายใจส่วนบน…"* — ตรงกับเอกสาร (`1.md` บรรทัด 139)

เพราะ improved ค้นเจอ chunk ที่มีคำตอบเป็นอันดับ 1 ส่วน baseline เจอเป็นอันดับ 4 (คำถามที่ 3 ในชุดวัดผล)

### ข้อจำกัด

- ชุดทดสอบเล็ก (14 คำถาม) และตัดสินว่า "เจอ" ด้วยการหาวลีคำตอบใน chunk — chunk ใหญ่ของ baseline มีโอกาสมีวลีอยู่มากกว่า การเทียบนี้จึงเอียงเข้าข้าง baseline อยู่แล้ว
- improved ช้ากว่า/แพงกว่าตอนนำเข้า: เรียก LLM 212 ครั้งด้วยเอกสารเต็ม (~2.6 ล้าน token) แทน 53 ครั้งด้วย chunk (~0.1 ล้าน token) — แต่ตอนถามตอบส่งข้อความให้ LLM น้อยกว่า 4.3 เท่าทุกคำถาม และมี cache ไม่ต้องจ่ายซ้ำ
- ใช้ Qwen 14B (ไม่ใช่ 72B ตามสไลด์) — ตัวเลขจึงเทียบกับผลที่ใช้ 72B ตรง ๆ ไม่ได้ แต่ (1) กับ (2) ใช้ LLM ตัวเดียวกัน
- Qwen บางครั้งหลุดไปตอบเป็นภาษาจีน (โมเดลถูกเทรนด้วยภาษาจีนเยอะ ภาษาไทยน้อย) — เพิ่มการตรวจจับแล้วขอให้ตอบใหม่ในทั้ง 2 ระบบ; เหลือบริบทภาษาจีน 1/53 ใน baseline, 0/212 ใน improved
