"""
สำรอง / กู้คืน OpenSearch index ของทั้ง 2 ระบบ เป็นไฟล์ JSON ในโฟลเดอร์ results/indices/
ใช้ย้ายไปเครื่องอื่นโดยไม่ต้องรัน ingest ใหม่ (ไม่ต้องเรียก LLM / ไม่ต้อง embed)

  เครื่องที่ ingest แล้ว:  python index_backup.py export
  เครื่องใหม่:            python index_backup.py import
"""
import sys
import json
from pathlib import Path

from opensearchpy import OpenSearch, helpers

INDICES = ["anthropic-vector-index", "anthropic-bm25-index", "improved-vector-index", "improved-bm25-index"]
BACKUP_DIR = Path("results/indices")
# settings ที่ต้องใช้สร้าง index ใหม่ (ตัดค่าเฉพาะเครื่องเดิม เช่น uuid, creation_date ทิ้ง)
KEEP_SETTINGS = ("index.knn", "index.similarity", "index.analysis", "index.queries")

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)


def export():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in INDICES:
        settings = client.indices.get_settings(index=name, flat_settings=True)[name]["settings"]
        backup = {
            "settings": {k: v for k, v in settings.items() if k.startswith(KEEP_SETTINGS)},
            "mappings": client.indices.get_mapping(index=name)[name]["mappings"],
            "docs": [{"_id": h["_id"], **h["_source"]} for h in helpers.scan(client, index=name)],
        }
        path = BACKUP_DIR / f"{name}.json"
        path.write_text(json.dumps(backup, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(backup['docs'])} docs -> {path}")


def restore():
    for name in INDICES:
        backup = json.loads((BACKUP_DIR / f"{name}.json").read_text(encoding="utf-8"))
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
        client.indices.create(index=name, body={"settings": backup["settings"], "mappings": backup["mappings"]})
        helpers.bulk(client, ({"_index": name, **doc} for doc in backup["docs"]))
        client.indices.refresh(index=name)
        print(f"{name}: กู้คืน {len(backup['docs'])} docs")


if __name__ == "__main__":
    {"export": export, "import": restore}[sys.argv[1]]()
