"""
Part 3 - เปรียบเทียบการค้นคืน (retrieval) ของ baseline (part 1) กับ improved (part 2)

วัดเฉพาะขั้น "ค้นเอกสาร" (ไม่เรียก LLM ตอบ) ด้วยชุดคำถามที่รู้คำตอบ:
  ถ้า chunk ที่ค้นได้มี "วลีคำตอบ" (gold) อย่างน้อย 1 วลีอยู่ในเนื้อหา = ค้นเจอเอกสารที่ถูก

  Hit@5   : สัดส่วนคำถามที่ chunk ที่ถูกอยู่ใน 5 อันดับแรก
  MRR@5   : ค่าเฉลี่ยของ 1/อันดับ ของ chunk ที่ถูกอันแรก (อยู่อันดับ 1 = 1.0, อันดับ 2 = 0.5, ...)
  Hit@1   : สัดส่วนคำถามที่ chunk อันดับ 1 ถูกเลย
  ctx chars: ความยาวเฉลี่ย (ตัวอักษร) ของเนื้อหา 5 chunk ที่ส่งให้ LLM (ยิ่งน้อย = ประหยัด token, LLM สับสนน้อย)

วิธีรัน (หลังรัน ingest ของทั้ง 2 ระบบแล้ว):  python compare_retrieval.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "part1_baseline"))
sys.path.insert(0, str(ROOT / "part2_improved"))
from onlysearchAuthenticRAG import AuthenticSearchRAG   # noqa: E402
from onlysearchImprovedRAG import ImprovedSearchRAG     # noqa: E402

K = 5

# (คำถาม, [วลีที่ยอมรับว่าเป็นคำตอบ]) - วลีคัดลอกมาจาก corpus_input ตรง ๆ
# มีหลายวลีเมื่อคำตอบเขียนได้หลายแบบ (ชื่อไทย/อังกฤษ) หรือคำตอบเป็นรายการหลายข้อ
EVAL_SET = [
    # 1.md หัดเยอรมัน
    ("หัดเยอรมันมีระยะฟักตัวนานเท่าไร", ["12-24 วัน"]),
    ("ควรฉีดวัคซีน MMR เข็มแรกตอนเด็กอายุเท่าไร", ["9-12 เดือน"]),
    ("คนที่เคยฉีดวัคซีนหัดเยอรมันแล้วจะติดเชื้อซ้ำได้ไหม", ["ทางเดินหายใจส่วนบนเท่านั้น"]),
    ("ทารกที่แม่ติดหัดเยอรมันตอนตั้งครรภ์อาจมีความผิดปกติอะไรบ้าง", ["ผนังหัวใจรั่ว", "ทารกเจริญเติบโตช้า"]),
    # 2.md อหิวาตกโรค
    ("อหิวาตกโรคเกิดจากเชื้ออะไร", ["Vibrio cholerae", "วิบริโอคอเลอเร"]),
    ("อหิวาตกโรคมีระยะฟักตัวนานเท่าไร", ["24-48 ชั่วโมง"]),
    ("ภาวะแทรกซ้อนของอหิวาตกโรคมีอะไรบ้าง", ["ภาวะเลือดเป็นกรด"]),
    ("ผู้ป่วยอหิวาต์ที่ช็อกควรได้รับสารน้ำชนิดใด", ["ริงเกอร์แล็กเทต", "Ringer lactate"]),
    # 44.md ต้อกระจก
    ("โรคประจำตัวอะไรทำให้เป็นต้อกระจกก่อนวัย", ["ต่อมไทรอยด์ผิดปกติ"]),
    ("คนเป็นเบาหวานผ่าตัดต้อกระจกได้หรือไม่", ["ผู้เป็นเบาหวานสามารถผ่าตัดต่อกระจกได้"]),
    ("ใช้ยาอะไรนาน ๆ แล้วเสี่ยงเป็นต้อกระจก", ["สเตียรอยด์เป็นเวลานาน"]),
    # 5555.md กรดไหลย้อน
    ("ยาที่ใช้รักษากรดไหลย้อนในระยะแรกมีอะไรบ้าง", ["Ranitidine", "รานิทิดีน"]),
    ("เป็นกรดไหลย้อน กินข้าวเสร็จต้องรอนานแค่ไหนก่อนออกกำลังกาย", ["2-3 ชั่วโมงหลังการรับประทานอาหาร"]),
    ("ทำไมคนเป็นเบาหวานถึงเป็นกรดไหลย้อนได้", ["การเสื่อมของประสาทกระเพาะ"]),
]


def evaluate(name, rag):
    rows = []
    for question, golds in EVAL_SET:
        results = rag.hybrid_search(question, k=K)
        contents = [src.get("content", "") for _, _, src in results]
        rank = next((i for i, c in enumerate(contents, 1) if any(g in c for g in golds)), None)
        rows.append({"question": question, "gold": golds, "rank": rank,
                     "ctx_chars": sum(len(c) for c in contents)})
    n = len(rows)
    return {
        "system": name,
        f"hit@{K}": sum(r["rank"] is not None for r in rows) / n,
        f"mrr@{K}": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "ctx_chars": sum(r["ctx_chars"] for r in rows) / n,
        "per_question": rows,
    }


def main():
    reports = [
        evaluate("part1_baseline", AuthenticSearchRAG()),
        evaluate("part2_improved", ImprovedSearchRAG()),
    ]

    print("\n" + "=" * 70)
    print(f"{'คำถาม':<55} {'baseline':>8} {'improved':>8}")
    for b, i in zip(reports[0]["per_question"], reports[1]["per_question"]):
        print(f"{b['question'][:55]:<55} {str(b['rank'] or '-'):>8} {str(i['rank'] or '-'):>8}")
    print("-" * 70)
    for r in reports:
        print(f"{r['system']:<16} Hit@{K} = {r[f'hit@{K}']:.2f}   MRR@{K} = {r[f'mrr@{K}']:.2f}"
              f"   Hit@1 = {r['hit@1']:.2f}   ctx chars = {r['ctx_chars']:,.0f}")

    with open("retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print("\nบันทึกผลไปยัง retrieval_comparison.json")


if __name__ == "__main__":
    main()
