#!/usr/bin/env python
import json
from pathlib import Path


REPLACEMENTS = [
    ("qủa", "quả"),
    ("Qủa", "Quả"),
    ("gía", "giá"),
    ("Gía", "Giá"),
    ("tường đương", "tương đương"),
    ("Tường đương", "Tương đương"),
    ("tưong", "tương"),
    ("Tưong", "Tương"),
    ("đươc", "được"),
    ("Đươc", "Được"),
    ("đươợc", "được"),
    ("hỏi đap", "hỏi đáp"),
    ("kết qủa", "kết quả"),
    ("Kết qủa", "Kết quả"),
    ("số liêu", "số liệu"),
    ("thưc tế", "thực tế"),
    ("Thưc tế", "Thực tế"),
    ("chinh xác", "chính xác"),
    ("Chinh xác", "Chính xác"),
    ("câu hoi", "câu hỏi"),
    ("Câu hoi", "Câu hỏi"),
]

FORBIDDEN_CHARS = ["—", "“", "”", "…"]


def normalize_text(text):
    changed = False
    out = text
    for src, dst in REPLACEMENTS:
        if src in out:
            changed = True
            out = out.replace(src, dst)
    return out, changed


def has_known_error(text):
    return any(src in text for src, _ in REPLACEMENTS)


def valid_record(record):
    if not isinstance(record, dict):
        return False, "not_object"
    prompt = record.get("prompt")
    response = record.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return False, "missing_prompt_response"
    if has_known_error(prompt) or has_known_error(response):
        return False, "known_error_left"
    if len(response) < 20:
        return False, "response_too_short"
    if any(ch in prompt or ch in response for ch in FORBIDDEN_CHARS):
        return False, "forbidden_char"
    return True, ""


def process_file(src_path, dst_path):
    total = 0
    normalized = 0
    dropped = 0
    output = []
    examples = []
    drop_reasons = {}

    for line_no, line in enumerate(src_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        before = line
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            drop_reasons["json_parse"] = drop_reasons.get("json_parse", 0) + 1
            continue

        prompt, prompt_changed = normalize_text(record.get("prompt", ""))
        response, response_changed = normalize_text(record.get("response", ""))
        if isinstance(record, dict):
            record["prompt"] = prompt
            record["response"] = response
        changed = prompt_changed or response_changed
        if changed:
            normalized += 1
            after = json.dumps(record, ensure_ascii=False)
            if len(examples) < 10:
                examples.append({"file": str(src_path), "line": line_no, "before": before, "after": after})

        ok, reason = valid_record(record)
        if not ok:
            dropped += 1
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue

        # Validate serialized JSON before writing.
        serialized = json.dumps(record, ensure_ascii=False)
        json.loads(serialized)
        output.append(serialized)

    dst_path.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return {
        "input": str(src_path),
        "output": str(dst_path),
        "total": total,
        "normalized": normalized,
        "dropped": dropped,
        "final": len(output),
        "drop_reasons": drop_reasons,
        "examples": examples,
    }


def main():
    train = process_file(Path("data/nano2_b_train.jsonl"), Path("data/nano2_b_train_fixed.jsonl"))
    val = process_file(Path("data/nano2_b_val.jsonl"), Path("data/nano2_b_val_fixed.jsonl"))
    report = {
        "train": {k: v for k, v in train.items() if k != "examples"},
        "val": {k: v for k, v in val.items() if k != "examples"},
        "examples": (train["examples"] + val["examples"])[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if train["final"] < 200 else 0


if __name__ == "__main__":
    raise SystemExit(main())
