#!/usr/bin/env python
import argparse
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path


NORMALIZE_MAP = str.maketrans({
    "\u2014": "-",
    "\u2013": "-",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2026": "...",
})

SPACE_RE = re.compile(r"\s+")


def normalize_text(value):
    return str(value).translate(NORMALIZE_MAP)


def prompt_key(prompt):
    return SPACE_RE.sub("", prompt).lower()


def first_conversation_pair(conversations):
    if not isinstance(conversations, list):
        return None
    prompt = None
    response = None
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "user" and prompt is None:
            prompt = content
        elif role == "assistant" and prompt is not None:
            response = content
            break
    if prompt is None or response is None:
        return None
    return {"prompt": prompt, "response": response}


def convert_record(obj):
    if not isinstance(obj, dict):
        return None
    if "prompt" in obj and "response" in obj:
        return {"prompt": obj["prompt"], "response": obj["response"]}
    if "conversations" in obj:
        return first_conversation_pair(obj["conversations"])
    return None


def too_similar(key, kept_keys):
    if not key:
        return False
    for existing in kept_keys:
        short = min(len(key), len(existing))
        long = max(len(key), len(existing))
        if long == 0:
            continue
        if short / long <= 0.90:
            continue
        if key == existing:
            return True
        if SequenceMatcher(None, key, existing, autojunk=False).ratio() > 0.90:
            return True
    return False


def clean(args):
    input_dir = Path(args.input_dir)
    output_train = Path(args.train_output)
    output_val = Path(args.val_output)
    log_path = Path(args.log)

    stats = {
        "total_read": 0,
        "format_skipped": 0,
        "dedup_skipped": 0,
        "length_skipped": 0,
    }

    accepted = []
    kept_keys = []
    log_lines = []

    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                raw = line.strip()
                if not raw:
                    continue
                stats["total_read"] += 1
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    stats["format_skipped"] += 1
                    log_lines.append(f"{path}:{line_no}: invalid json: {exc}")
                    continue

                record = convert_record(obj)
                if record is None:
                    stats["format_skipped"] += 1
                    log_lines.append(f"{path}:{line_no}: unrecognized format")
                    continue

                prompt = normalize_text(record["prompt"])
                response = normalize_text(record["response"])

                if not prompt.strip() or len(response) < 10 or len(response) > 800:
                    stats["length_skipped"] += 1
                    continue

                key = prompt_key(prompt)
                if too_similar(key, kept_keys):
                    stats["dedup_skipped"] += 1
                    continue

                kept_keys.append(key)
                accepted.append({"prompt": prompt, "response": response})

    random.Random(42).shuffle(accepted)
    val_size = max(50, len(accepted) // 10)
    if val_size > len(accepted):
        raise RuntimeError(
            f"Not enough samples for a validation set of at least 50: {len(accepted)}"
        )
    train_size = len(accepted) - val_size
    train = accepted[:train_size]
    val = accepted[train_size:]

    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_val.parent.mkdir(parents=True, exist_ok=True)
    with output_train.open("w", encoding="utf-8", newline="\n") as fh:
        for item in train:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    with output_val.open("w", encoding="utf-8", newline="\n") as fh:
        for item in val:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")

    stats["train"] = len(train)
    stats["val"] = len(val)
    stats["final"] = len(accepted)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Clean Nano 1 SFT data")
    parser.add_argument("--input-dir", default="datasach")
    parser.add_argument("--train-output", default="data/nano1_train.jsonl")
    parser.add_argument("--val-output", default="data/nano1_val.jsonl")
    parser.add_argument("--log", default="data/nano1_clean_skipped.log")
    clean(parser.parse_args())


if __name__ == "__main__":
    main()
