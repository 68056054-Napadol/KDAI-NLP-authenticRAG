# อธิบายการทำงาน: Baseline vs Improved

เอกสารนี้อธิบายว่าระบบ (1) baseline และ (2) improved ทำงานอย่างไร, improved เปลี่ยน/เพิ่มอะไร, อยู่ตรงไหนของโค้ด และแต่ละจุดทำให้ผลดีขึ้นอย่างไร พร้อมผลวัดจริง

> ผลทั้งหมดในเอกสารนี้วัดด้วย LLM **Qwen 2.5 14B ผ่าน Ollama** (ทั้ง 2 ระบบใช้ตัวเดียวกัน) และชุดทดสอบ 14 คำถามใน [compare_retrieval.py](compare_retrieval.py#L28)
> ความหมายของตัวชี้วัด:
> - **Hit@1** = สัดส่วนคำถามที่ chunk อันดับ 1 มีคำตอบ
> - **MRR@5** = ค่าเฉลี่ยของ 1/อันดับ ของ chunk ที่มีคำตอบ (อันดับ 1 = 1.0, อันดับ 2 = 0.5, …) ยิ่งใกล้ 1 ยิ่งดี

---

## สารบัญ

1. [ภาพรวม](#1-ภาพรวม)
2. [Baseline ทำงานอย่างไร](#2-baseline-ทำงานอย่างไร)
3. [Improved ทำงานอย่างไร](#3-improved-ทำงานอย่างไร)
4. [สิ่งที่เปลี่ยน 5 จุด: โค้ดตรงไหน ดีขึ้นอย่างไร](#4-สิ่งที่เปลี่ยน-5-จุด)
5. [สิ่งที่เพิ่มเพื่อให้ใช้งานได้จริง (ไม่กระทบความแม่นยำ)](#5-สิ่งที่เพิ่มเพื่อให้ใช้งานได้จริง)
6. [สรุปผล](#6-สรุปผล)

---

## 1. ภาพรวม

ทั้ง 2 ระบบเป็น RAG แบบ **Contextual Retrieval** (สไลด์หน้า 20–25) แบ่งเป็น 2 ขั้น

```
ขั้นนำเข้า (ingest)  : เอกสาร → หั่นเป็น chunk → LLM เขียน "บริบท" ให้แต่ละ chunk → เก็บลง OpenSearch 2 index
ขั้นค้นหา + ตอบ      : คำถาม → ค้นใน 2 index (BM25 + vector) → รวมผล → ส่ง chunk ให้ LLM ตอบ
```

| | Baseline (1) | Improved (2) |
|---|---|---|
| ไฟล์ ingest | [part1_baseline/authenticRAG.py](part1_baseline/authenticRAG.py) | [part2_improved/improvedRAG.py](part2_improved/improvedRAG.py) |
| ไฟล์ค้นหา + ตอบ | [part1_baseline/onlysearchAuthenticRAG.py](part1_baseline/onlysearchAuthenticRAG.py) | [part2_improved/onlysearchImprovedRAG.py](part2_improved/onlysearchImprovedRAG.py) |
| index | `anthropic-vector-index`, `anthropic-bm25-index` | `improved-vector-index`, `improved-bm25-index` |

**สิ่งที่ไม่ได้เปลี่ยน** (เพื่อให้เทียบกันได้ยุติธรรม):
- LLM ตัวเดียวกัน ทั้งตอนเขียนบริบทและตอนตอบคำถาม
- Prompt ตอบคำถามสุดท้าย ([generate_response](part1_baseline/onlysearchAuthenticRAG.py#L223) — improved สืบทอดไปใช้ตรง ๆ)
- Embedding model `BAAI/bge-m3`
- สูตร RRF ([rrf_fusion](part1_baseline/onlysearchAuthenticRAG.py#L125) — improved สืบทอดไปใช้ตรง ๆ)
- จำนวน chunk ที่ส่งให้ LLM (5 อัน)

---

## 2. Baseline ทำงานอย่างไร

โค้ดจากสไลด์หน้า 22 (Hands-on 1) และหน้า 25 (Hands-on 2)

### 2.1 ขั้นนำเข้า — [authenticRAG.py](part1_baseline/authenticRAG.py)

```
corpus_input/*.md
   │  load_documents()                ← TextLoader อ่านไฟล์ → MarkdownNodeParser หั่นตามหัวข้อ
   ▼
53 chunk (ยาว 112–8,404 ตัวอักษร)
   │  add_documents_with_context()    ← วนทีละ chunk
   │     ├─ generate_context()        ← LLM เขียนบริบท (ส่ง chunk เป็น "เอกสาร")
   │     ├─ embed_query(chunk+บริบท)  ← bge-m3 → vector 1,024 มิติ
   │     ├─ vector index ← {embedding, content}
   │     └─ BM25 index   ← {content, contextualized_content}  (analyzer: standard)
   ▼
OpenSearch: anthropic-vector-index, anthropic-bm25-index
```

| ขั้นตอน | โค้ด |
|---|---|
| สร้าง index (vector + BM25 แบบ `standard`) | [`_create_or_update_indices`](part1_baseline/authenticRAG.py#L84) |
| หั่น chunk | [`MarkdownNodeParser(chunk_size=256)`](part1_baseline/authenticRAG.py#L79), [`load_documents`](part1_baseline/authenticRAG.py#L152) |
| prompt สร้างบริบท | [`get_context_prompt`](part1_baseline/authenticRAG.py#L193) |
| เรียก LLM สร้างบริบท | [`generate_context`](part1_baseline/authenticRAG.py#L208) |
| วนนำเข้าแต่ละ chunk | [`add_documents_with_context`](part1_baseline/authenticRAG.py#L214) |

### 2.2 ขั้นค้นหา + ตอบ — [onlysearchAuthenticRAG.py](part1_baseline/onlysearchAuthenticRAG.py)

```
คำถาม
   ├─ sparse_search()  BM25 top 10
   ├─ dense_search()   vector top 10
   ▼
rrf_fusion()  รวม 2 รายการด้วยสูตร 1/(60 + rank)  →  top 5
   ▼
search_for_question()  ต่อ "Content + Context" ของ 5 chunk เป็นข้อความเดียว
   ▼
generate_response()  LLM ตอบ
```

| ขั้นตอน | โค้ด |
|---|---|
| BM25 (multi_match ที่ `content` + `contextualized_content`) | [`sparse_search`](part1_baseline/onlysearchAuthenticRAG.py#L86) |
| Vector kNN | [`dense_search`](part1_baseline/onlysearchAuthenticRAG.py#L104) |
| รวมผล RRF | [`rrf_fusion`](part1_baseline/onlysearchAuthenticRAG.py#L125) |
| hybrid (ดึงอย่างละ k×2 = 10 แล้วตัดเหลือ 5) | [`hybrid_search`](part1_baseline/onlysearchAuthenticRAG.py#L143) |
| ประกอบ context ส่งให้ LLM | [`search_for_question`](part1_baseline/onlysearchAuthenticRAG.py#L172) |
| prompt ตอบคำถาม | [`generate_response`](part1_baseline/onlysearchAuthenticRAG.py#L223) |

---

## 3. Improved ทำงานอย่างไร

### 3.1 ขั้นนำเข้า — [improvedRAG.py](part2_improved/improvedRAG.py)

```
corpus_input/*.md
   │  load_documents()                       ← [1] หั่นตามหัวข้อ แล้วตัดซ้ำให้ ≤512 token (overlap 64)
   ▼
212 chunk (ยาว 112–728 ตัวอักษร)
   │  add_documents()
   │     ├─ อ่าน cache → เรียก LLM เฉพาะ chunk ที่ยังไม่มีบริบท        (เพิ่ม: cache)
   │     ├─ generate_context() แบบขนาน                               ← [2] ส่ง "เอกสารเต็ม" + ขอตอบภาษาไทย
   │     ├─ encode(บริบท+chunk) ทีเดียวทั้ง batch บน GPU
   │     ├─ vector index ← {embedding, content, contextualized_content, source}   ← [3]
   │     └─ BM25 index   ← {content, contextualized_content, source}  (analyzer: thai)  ← [4]
   ▼
OpenSearch: improved-vector-index, improved-bm25-index
```

### 3.2 ขั้นค้นหา + ตอบ — [onlysearchImprovedRAG.py](part2_improved/onlysearchImprovedRAG.py)

`ImprovedSearchRAG` **สืบทอด** (`class ImprovedSearchRAG(AuthenticSearchRAG)`, [บรรทัด 25](part2_improved/onlysearchImprovedRAG.py#L25)) จาก baseline — ใช้ BM25, vector, RRF, prompt ตอบคำถามของ baseline ทั้งหมด เปลี่ยนแค่ 2 อย่าง:

1. ชี้ไปที่ index ใหม่ ([บรรทัด 28–29](part2_improved/onlysearchImprovedRAG.py#L28))
2. override `hybrid_search` เพื่อเพิ่ม reranking ([บรรทัด 33–44](part2_improved/onlysearchImprovedRAG.py#L33)) ← [5]

```
คำถาม
   ├─ sparse_search()  BM25 top 40      (ของ baseline, แต่ index ตัดคำไทยแล้ว)
   ├─ dense_search()   vector top 40    (ของ baseline)
   ▼
rrf_fusion()  →  top 20                 (ของ baseline)
   ▼
reranker (bge-reranker-v2-m3) ให้คะแนนใหม่  →  top 5     ← [5] เพิ่มใหม่
   ▼
search_for_question() + generate_response()               (ของ baseline)
```

---

## 4. สิ่งที่เปลี่ยน 5 จุด

### [1] Chunking — หั่นให้ขนาดสม่ำเสมอ

| | โค้ด |
|---|---|
| Baseline | [`MarkdownNodeParser(chunk_size=256)`](part1_baseline/authenticRAG.py#L79) |
| Improved | [`MarkdownNodeParser()`](part2_improved/improvedRAG.py#L87) + [`SentenceSplitter(chunk_size=512, chunk_overlap=64)`](part2_improved/improvedRAG.py#L88), ใช้ใน [`load_documents`](part2_improved/improvedRAG.py#L120) |

**ปัญหาของ baseline:** `chunk_size` **ไม่ใช่พารามิเตอร์ของ `MarkdownNodeParser`** (parser นี้หั่นตามหัวข้ออย่างเดียว) ค่า 256 จึงถูกเมินโดยไม่มี error → 1 หัวข้อ = 1 chunk ไม่ว่าจะยาวแค่ไหน

| | จำนวน chunk | สั้นสุด | ค่ากลาง | ยาวสุด |
|---|---|---|---|---|
| Baseline | 53 | 112 | 1,188 | **8,404** ตัวอักษร |
| Improved | 212 | 112 | 472 | **728** ตัวอักษร |

**ทำไมดีขึ้น:**
- chunk ยาว 5,000–8,000 ตัวอักษรถูกบีบเป็น vector **เดียว** ความหมายของเรื่องย่อยข้างในจึง "เจือจาง"
- ตอนตอบ ต้องส่งเนื้อหาที่ไม่เกี่ยวข้องไปให้ LLM ด้วยเยอะ

**ผล:**
- ข้อความที่ส่งให้ LLM ต่อคำถาม: **10,257 → 2,408 ตัวอักษร (น้อยลง 4.3 เท่า)**
- ตัวอย่าง — *"อหิวาตกโรคมีระยะฟักตัวนานเท่าไร"* (คำตอบ: 24-48 ชั่วโมง)
  - Baseline: คำตอบอยู่ใน chunk `## สาเหตุของอหิวาตกโรค` ยาว **5,600 ตัวอักษร** (ข้อมูลระยะฟักตัวจมอยู่ในหัวข้อ "สาเหตุ") → ได้อันดับ **3**
  - Improved: chunk `2.md#78` ยาว 472 ตัวอักษร ที่พูดเรื่องระยะฟักตัวโดยเฉพาะ → ได้อันดับ **1**

**ข้อควรระวัง:** chunk เล็ก "ขาดบริบท" ง่าย — ถ้าหั่นเล็กอย่างเดียวโดยไม่มีบริบท vector search **แย่ลง** (MRR 0.86 → 0.75, ดู [ablation](#ablation-แยกผลทีละส่วน)) จึงต้องใช้คู่กับจุด [2]

---

### [2] Contextual Retrieval — ส่งเอกสารเต็มให้ LLM

| | โค้ด |
|---|---|
| Baseline | [`full_doc_content = doc.page_content`](part1_baseline/authenticRAG.py#L227) แล้วส่งเข้า [`generate_context`](part1_baseline/authenticRAG.py#L230) |
| Improved | [`CONTEXT_PROMPT`](part2_improved/improvedRAG.py#L56) + [`generate_context`](part2_improved/improvedRAG.py#L155) ที่รับ `full_doc` (เอกสารเต็มจาก [`load_documents`](part2_improved/improvedRAG.py#L127)) |

**ปัญหาของ baseline (บั๊ก):** ตัวแปรชื่อ `full_doc_content` แต่จริง ๆ คือ **chunk ตัวเอง** prompt ที่ส่งไปจึงเป็น

```
<document> chunk A </document>          ← ควรเป็นเอกสารเต็ม
<chunk> chunk A </chunk>
อธิบายว่า chunk นี้อยู่ตรงไหนของเอกสาร
```

LLM ไม่เคยเห็นเอกสารเต็ม จึงทำได้แค่ **สรุป chunk ซ้ำ** ไม่ได้เพิ่มข้อมูลใหม่

**Improved:** ส่งเอกสารเต็ม (~10,000–13,400 token) + chunk และเพิ่มคำสั่ง *"Write the context in the same language as the document (Thai)"* ([บรรทัด 65](part2_improved/improvedRAG.py#L65)) เพื่อให้บริบทเป็นภาษาไทย (BM25 จะได้ใช้ประโยชน์จากบริบทด้วย)

**ตัวอย่างจริง** — chunk `2.md#96`:

```
chunk  : (Ringer lactate) หรืออะซีทาร์ (Acetar) แต่ถ้าไม่มีอาจใช้น้ำเกลือนอร์มัล (NSS) แทน ...
         ↑ ตัว chunk ไม่ได้บอกเลยว่าเป็นโรคอะไร
บริบท  : ในส่วนของการรักษาผู้ป่วยที่มีอาการรุนแรงของอหิวาตกโรค โดยแพทย์จะให้สารน้ำและเกลือแร่
         ทางหลอดเลือดดำ และอาจให้ยาปฏิชีวนะเพื่อฆ่าเชื้อแบคทีเรีย Vibrio cholerae
```

**ผล (vector search อย่างเดียว, MRR@5):**

| | ไม่มีบริบท | มีบริบท | เปลี่ยนแปลง |
|---|---|---|---|
| Baseline | 0.86 | 0.86 | **±0** ← บริบทไม่ช่วยเลย (ยืนยันบั๊ก) |
| Improved | 0.75 | **0.90** | **+0.15** ← บริบทจากเอกสารเต็มช่วยจริง |

**ทำไมดีขึ้น:** ตรงกับแนวคิดของ Anthropic (สไลด์หน้า 18–19) — บริบทบอกว่า chunk นี้ "เป็นของเอกสารไหน หัวข้อไหน" ทำให้ทั้ง embedding และ BM25 จับคู่กับคำถามได้ แม้ตัว chunk จะไม่มีคำสำคัญอยู่เลย

---

### [3] Vector index เก็บบริบทด้วย

| | โค้ด |
|---|---|
| Baseline | [`vector_data = {embedding, doc_id, content, chunk_id}`](part1_baseline/authenticRAG.py#L246) — **ไม่มี** `contextualized_content` |
| Improved | [`fields = {content, contextualized_content, source}`](part2_improved/improvedRAG.py#L199) ใส่ทั้ง 2 index ([บรรทัด 200–201](part2_improved/improvedRAG.py#L200)) |

**ปัญหาของ baseline:** [`rrf_fusion`](part1_baseline/onlysearchAuthenticRAG.py#L132) เก็บข้อมูลของ chunk จาก ranker ที่เจอ chunk นั้น**ก่อน** ถ้า chunk เจอจาก vector search อย่างเดียว (ซึ่งเกิดบ่อยมาก เพราะ BM25 ของ baseline ใช้ไม่ได้ ดูจุด [4]) ข้อมูลที่ได้มาจาก vector index ซึ่ง**ไม่มีบริบท** → [`search_for_question`](part1_baseline/onlysearchAuthenticRAG.py#L213) ส่ง `Context: ` ว่างให้ LLM

**ตัวอย่างจริง** — *"คนที่เคยฉีดวัคซีนหัดเยอรมันแล้วจะติดเชื้อซ้ำได้ไหม"*

| อันดับ | Baseline | Improved |
|---|---|---|
| 1 | `doc_13` วิธีป้องกันโรคหัดเยอรมัน — **บริบท: (ว่าง)** | ✔ `1.md#45` — บริบท: *"…การอธิบายเกี่ยวกับภูมิคุ้มกันของผู้ที่เคย…"* |
| 2 | `doc_2` สาเหตุของโรคหัดเยอรมัน — **บริบท: (ว่าง)** | `1.md#53` — บริบท: *"ในส่วนของการป้องกันโรคหัดเยอรมัน…"* |
| 3 | `doc_11` ควรไปพบแพทย์เมื่อ — **บริบท: (ว่าง)** | `1.md#17` — บริบท: *"…ภาวะแทรกซ้อนที่อาจเกิดขึ้น…"* |

Baseline: chunk ที่มีคำตอบอยู่อันดับ **4** และทั้ง 3 อันดับแรกส่งให้ LLM โดยไม่มีบริบท
Improved: อันดับ **1** และทุก chunk มีบริบท

---

### [4] BM25 ตัดคำภาษาไทย (สำคัญที่สุด)

| | โค้ด |
|---|---|
| Baseline | [`"analyzer": "standard"`](part1_baseline/authenticRAG.py#L118) |
| Improved | [`"analyzer": "thai"`](part2_improved/improvedRAG.py#L112) (มีในตัว OpenSearch อยู่แล้ว) |

**ปัญหาของ baseline:** `standard` analyzer **ไม่รู้จักการตัดคำไทย** (ภาษาไทยไม่เว้นวรรคระหว่างคำ) จึงเก็บทั้งวลีเป็น token เดียว

```
"โรคหัดเยอรมันมีอันตรายกับหญิงตั้งครรภ์"
standard → [โรคหัดเยอรมันมีอันตรายกับหญิงตั้งครรภ์]          ← 1 token
thai     → [โรค] [หัดเยอรมัน] [อันตราย] [หญิง] [ครรภ์]     ← 5 token
```

BM25 ให้คะแนนจาก term ที่ตรงกัน (TF/DF ใน inverted index, สไลด์หน้า 24) ถ้า term คือ "ทั้งวลี" ก็แทบไม่มีทางตรงกับคำถาม

**ผล:**

| | Baseline | Improved |
|---|---|---|
| BM25 คืนผลลัพธ์ได้ (จาก 14 คำถาม) | **2** | **14** |
| BM25 อย่างเดียว MRR@5 | **0.07** | **0.74** |

**ผลกระทบต่อ baseline ทั้งระบบ:** "hybrid search" ของ baseline แท้จริงคือ **vector search อย่างเดียว** และการรวม BM25 ที่ไร้ประโยชน์ด้วย RRF ยัง**ดึงผลลง** (vector 0.86 → hybrid 0.81) — และยังเป็นต้นเหตุของปัญหาในจุด [3]

---

### [5] Reranking ด้วย cross-encoder (เพิ่มใหม่)

| | โค้ด |
|---|---|
| Baseline | ไม่มี — ใช้ top 5 จาก RRF เลย ([`hybrid_search`](part1_baseline/onlysearchAuthenticRAG.py#L143)) |
| Improved | [`RERANK_MODEL`, `NUM_CANDIDATES = 20`](part2_improved/onlysearchImprovedRAG.py#L21), override [`hybrid_search`](part2_improved/onlysearchImprovedRAG.py#L33) |

```python
candidates = super().hybrid_search(query, k=20)              # hybrid เดิมของ baseline → 20 อันดับ
pairs = [(query, บริบท + chunk) for ... in candidates]
scores = self.reranker.predict(pairs)                        # cross-encoder ให้คะแนนใหม่
return sorted(...)[:5]                                       # เหลือ 5
```

| | Bi-encoder (bge-m3) | Cross-encoder (bge-reranker-v2-m3) |
|---|---|---|
| วิธีทำงาน | แปลงคำถาม/เอกสาร **แยกกัน** แล้ววัดระยะ vector | อ่าน **คำถาม + เอกสารพร้อมกัน** แล้วให้คะแนน |
| ความเร็ว | เร็ว (vector เอกสารคำนวณไว้ตอน ingest) | ช้า (คำนวณใหม่ทุกคู่) |
| ความแม่น | หยาบ | แม่นกว่า |

**ทำไมดีขึ้น:** ค้นกว้างด้วยวิธีที่เร็ว (20 อันดับ) แล้วจัดลำดับใหม่ด้วยวิธีที่แม่น นอกจากนี้ RRF ใช้แค่ "อันดับ" ไม่สนเนื้อหา — reranker ตัดสินจากเนื้อหาจริง

**ผล:** improved hybrid MRR@5 **0.89 → 0.96** หลัง rerank

**ตัวอย่างจริง** — *"ใช้ยาอะไรนาน ๆ แล้วเสี่ยงเป็นต้อกระจก"* (คำตอบ: สเตียรอยด์)

| อันดับ | Baseline | Improved |
|---|---|---|
| 1 | `doc_29` สาเหตุของต้อกระจก (2,065 ตัวอักษร, เรื่องทั่วไป) | `44.md#123` *"…ยาเพรดนิโซโลน (Prednisolone)…"* — บริบท: *"…การใช้ยาสเตียรอยด์เป็นเวลานาน ๆ อาจทำให้เกิดต้อกระจก"* |
| 2 | `doc_31` การวินิจฉัยต้อกระจก | ✔ `44.md#152` วิธีป้องกันต้อกระจก (มีวลี "สเตียรอยด์เป็นเวลานาน") |
| … | ✔ อันดับ **5** | |

(อันดับ 1 ของ improved ก็ตอบถูก — เป็นเรื่องยาสเตียรอยด์ — แต่วลีที่ใช้ตัดสินในชุดวัดผลอยู่ใน chunk อันดับ 2 ตัวเลขของ improved ข้อนี้จึง "ต่ำกว่าความจริง")

---

## 5. สิ่งที่เพิ่มเพื่อให้ใช้งานได้จริง

สิ่งเหล่านี้**ไม่ได้ทำให้ค้นแม่นขึ้น** แต่ทำให้รันได้สำเร็จ ประหยัด และตรวจสอบได้

| สิ่งที่เพิ่ม | โค้ด | ทำไมต้องมี |
|---|---|---|
| **Cache บริบท** | [`CONTEXT_CACHE`](part2_improved/improvedRAG.py#L54), [`add_documents`](part2_improved/improvedRAG.py#L177) | ขั้นสร้างบริบทของ improved แพง (~2.6 ล้าน token) ระหว่างทำ credit หมด — cache ทำให้รันต่อจากที่ค้างได้ ไม่จ่ายซ้ำ |
| **เรียก LLM ขนาน + retry** | [`ThreadPoolExecutor`](part2_improved/improvedRAG.py#L183), [`generate_context`](part2_improved/improvedRAG.py#L155) | 212 ครั้งแบบทีละครั้งช้ามาก / รับมือ rate limit (รอ 5, 10, 20, 40 วินาที) |
| **ตรวจภาษาจีน แล้วขอให้ตอบใหม่** | baseline: [`CHINESE`](part1_baseline/authenticRAG.py#L23), [`call_qwen_api`](part1_baseline/authenticRAG.py#L129) · improved: [`generate_context`](part2_improved/improvedRAG.py#L161) | Qwen (โมเดลจีน) บางครั้งหลุดไปตอบภาษาจีน — เดิมมีบริบทภาษาจีน 37/212 → เหลือ **0/212** (baseline 6 → 1/53) ใส่ทั้ง 2 ระบบเหมือนกัน |
| **รัน LLM บนเครื่อง (Ollama)** | `LLM_PROVIDER=ollama` ใน [baseline](part1_baseline/authenticRAG.py#L28) และ [improved](part2_improved/improvedRAG.py#L47), [Modelfile](Modelfile) | ไม่ต้องใช้ credit — Modelfile ขยาย context เป็น 16k token (ค่าเริ่มต้น 4,096 จะตัดเอกสารทิ้งเงียบ ๆ) |
| **แสดงโมเดลที่ใช้** | print ใน `__init__` ของทุกสคริปต์ | รู้ทันทีว่ากำลังใช้ LLM ตัวไหน รันบน GPU หรือไม่ |
| **สำรอง/กู้คืน index** | [index_backup.py](index_backup.py) | ย้ายไปเครื่องอื่นได้โดยไม่ต้อง ingest ใหม่ |
| **ชุดวัดผล** | [compare_retrieval.py](compare_retrieval.py) | วัด Hit@5 / MRR@5 / Hit@1 ของทั้ง 2 ระบบด้วยคำถามชุดเดียวกัน |

แก้ไขในโค้ด baseline เฉพาะส่วนที่จำเป็นต่อการรัน (Ollama, ตรวจภาษาจีน, print) และติด comment `[เพิ่มจากโค้ดในสไลด์]` ไว้ทุกจุด — ถ้าไม่ตั้ง `LLM_PROVIDER` และ LLM ไม่ตอบภาษาจีน baseline ทำงานเหมือนโค้ดในสไลด์ทุกอย่าง

---

## 6. สรุปผล

### ผลรวม (14 คำถาม, [retrieval_comparison.json](retrieval_comparison.json))

| | Baseline | Improved |
|---|---|---|
| Hit@5 | 1.00 | 1.00 |
| **MRR@5** | 0.81 | **0.96** |
| **Hit@1** | 0.71 (10/14) | **0.93** (13/14) |
| ข้อความส่งให้ LLM ต่อคำถาม | 10,257 ตัวอักษร | **2,408** ตัวอักษร |

อันดับของ chunk ที่มีคำตอบ ในแต่ละคำถาม:

```
baseline : 1  1  4  1  1  3  1  1  1  1  5  1  1  2
improved : 1  1  1  1  1  1  1  1  1  1  2  1  1  1
```

### Ablation (แยกผลทีละส่วน)

MRR@5 — "ไม่มีบริบท" คือรอบแรกที่ยังไม่ได้ใช้ LLM สร้างบริบท ([results/retrieval_no_llm_context.json](results/retrieval_no_llm_context.json))

| วิธีค้น | Baseline ไม่มีบริบท | Baseline มีบริบท | Improved ไม่มีบริบท | Improved มีบริบท |
|---|---|---|---|---|
| BM25 อย่างเดียว | 0.07 | 0.07 | 0.71 | 0.74 |
| Vector อย่างเดียว | 0.86 | 0.86 | 0.75 | **0.90** |
| Hybrid (RRF) | 0.82 | 0.81 | 0.77 | 0.89 |
| Hybrid + rerank | – | – | 0.87 | **0.96** |

### แต่ละจุดช่วยอะไร

| จุด | ช่วยอะไร | หลักฐาน |
|---|---|---|
| [4] Thai analyzer | ทำให้ BM25 ใช้งานได้ | BM25 MRR 0.07 → 0.71+ |
| [1] Chunk ≤512 token | ส่งข้อความให้ LLM น้อยลง, chunk ตรงประเด็น | 10,257 → 2,408 ตัวอักษร; แต่ถ้าไม่มี [2] vector แย่ลง 0.86 → 0.75 |
| [2] บริบทจากเอกสารเต็ม | แก้ปัญหา chunk เล็กขาดบริบท | vector 0.75 → 0.90 (baseline ±0) |
| [3] เก็บบริบทใน vector index | ทุก chunk ที่ส่งให้ LLM มีบริบท | ตัวอย่างคำถามวัคซีน: baseline 3 อันดับแรกไม่มีบริบท |
| [5] Reranking | จัดอันดับให้ chunk ที่ถูกขึ้นมาอันดับ 1 | hybrid 0.89 → 0.96 |

### ตัวอย่างคำตอบสุดท้าย

*"ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?"* ([authentic](authentic_rag_search_results.json) / [improved](improved_rag_search_results.json))

- **Baseline:** *"…มีโอกาสติดเชื้อน้อยมากหรือแทบไม่มีเลย … ป้องกันได้เกือบ 100%…"* — ความรู้ทั่วไป ไม่ได้มาจากเอกสาร
- **Improved:** *"…มักจะมีภูมิคุ้มกันตลอดชีวิต … แต่ยังมีความเป็นไปได้ว่าการได้รับเชื้อใหม่อาจทำให้ติดเชื้อได้ แต่จะไม่แสดงอาการและเชื้อจะอยู่แค่ที่ทางเดินหายใจส่วนบน…"* — ตรงกับเอกสาร `1.md` บรรทัด 139

### ราคาที่ต้องจ่าย

| | Baseline | Improved |
|---|---|---|
| ingest (จ่ายครั้งเดียว) | 53 ครั้ง × chunk ×2 ≈ **0.1 ล้าน token** | 212 ครั้ง × เอกสารเต็ม ≈ **2.6 ล้าน token** (~25 เท่า) |
| ถามตอบ (ทุกคำถาม) | ~10,300 ตัวอักษร | **~2,400 ตัวอักษร** (ถูกกว่า 4.3 เท่า) |
| เวลาค้นหา | เร็วกว่า | ช้ากว่าเล็กน้อย (reranker 20 คู่) |

Improved ลงทุนครั้งเดียวตอนนำเข้า เพื่อให้ทุกคำถามหลังจากนั้นแม่นขึ้นและถูกลง (และใช้ cache / Ollama เพื่อลดต้นทุนนำเข้าได้)

### ข้อจำกัด

- ชุดทดสอบเล็ก (14 คำถาม) และตัดสินว่า "เจอ" ด้วยการหาวลีคำตอบใน chunk — chunk ใหญ่ของ baseline มีโอกาสมีวลีอยู่มากกว่า การเทียบนี้จึงเอียงเข้าข้าง baseline
- ใช้ Qwen 14B (ไม่ใช่ 72B ตามสไลด์) เพราะ credit หมด — ตัวเลขเทียบกับผลที่ใช้ 72B ตรง ๆ ไม่ได้ แต่ (1) กับ (2) ใช้ LLM ตัวเดียวกัน
