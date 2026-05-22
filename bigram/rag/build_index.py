"""Build a JSONL lexical RAG index from local files."""

import json
from pathlib import Path

from .chunker import chunk_text


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "source": obj.get("source", str(path)),
                "title": obj.get("title", Path(obj.get("source", str(path))).stem),
                "url": obj.get("url", ""),
                "text": obj.get("text", ""),
            })
    return rows


def build_index(input_dir, output_path):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".jsonl"}:
            continue
        if path.suffix.lower() == ".jsonl":
            documents = _read_jsonl(path)
        else:
            documents = [{
                "source": str(path),
                "title": path.stem,
                "url": "",
                "text": path.read_text(encoding="utf-8", errors="replace"),
            }]
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(chunk_text(doc["text"])):
                rows.append({
                    "id": f"{path.stem}:{doc_idx}:{chunk_idx}",
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": chunk,
                })
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"chunks": len(rows), "output": str(output_path)}
