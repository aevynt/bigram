#!/usr/bin/env python
import argparse
import json
import re
import sys

sys.path.insert(0, ".")

import torch

from scripts.chat_nano1 import generate_response, load_model
from bigram import BigramTokenizer


TESTS = [
    ("[TOOLS] search [/TOOLS] User: giá vàng hôm nay?", "search"),
    ("[TOOLS] search [/TOOLS] User: thời tiết Hà Nội ngày mai?", "search"),
    ("[TOOLS] calculate [/TOOLS] User: 15% của 2 triệu là bao nhiêu?", "calculate"),
    ("[TOOLS] search, calculate [/TOOLS] User: nếu vàng tăng 2% thì giá mới là bao nhiêu?", "search"),
    ("[TOOLS] translate [/TOOLS] User: dịch \"hello world\" sang tiếng Việt", "translate"),
    ("[TOOLS] search [/TOOLS] User: tỷ giá USD hôm nay?", "search"),
    ("[TOOLS] search, get_weather [/TOOLS] User: trời Đà Nẵng có mưa không?", "get_weather"),
    ("[NO TOOLS] User: bạn là ai?", None),
    ("[NO TOOLS] User: xin chào!", None),
    ("[NO TOOLS] User: 2 + 2 bằng mấy?", None),
    ("[TOOLS] summarize [/TOOLS] User: tóm tắt đoạn văn ngắn này", "summarize"),
    ("[NO TOOLS] User: thủ đô Việt Nam là gì?", None),
    ("[TOOLS] search [/TOOLS] User: số điện thoại của tôi là gì?", None),
    ("[TOOLS] call_api [/TOOLS] User: lấy danh sách sản phẩm từ API", "call_api"),
    ("[NO TOOLS] User: tạm biệt", None),
]


def parse_tool_call(response):
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.S)
    if not match:
        return None, False
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, False
    return parsed, True


def is_rubbish(text):
    bad_fragments = ["hink>", "th c", "khồngChao", "suan", "sẹa", "?", "�"]
    return any(fragment in text for fragment in bad_fragments) or text.strip() in {"<think>", ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="nano1/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)

    failures = []
    rows = []
    for index, (prompt, expected_tool) in enumerate(TESTS, 1):
        torch.manual_seed(2000 + index)
        response = generate_response(model, tokenizer, prompt, args, device)
        parsed, parse_ok = parse_tool_call(response)
        tool_name = parsed.get("name") if parsed else None
        ok = True
        reason = ""
        if expected_tool:
            ok = parse_ok and tool_name == expected_tool
            reason = "" if ok else f"expected tool {expected_tool}, got {tool_name}, parse_ok={parse_ok}"
        elif index == 13:
            ok = "<tool_call>" not in response and ("không" in response.lower() or "không biết" in response.lower())
            reason = "" if ok else "expected no tool and refusal"
        elif index in {8, 9, 15}:
            ok = ("Bigram Nano 2" in response and "Aevynt Lab" in response)
            reason = "" if ok else "expected Nano 2 identity response"
        elif index == 10:
            ok = "<tool_call>" not in response and "4" in response
            reason = "" if ok else "expected direct arithmetic answer"
        elif index == 12:
            ok = "<tool_call>" not in response and "Hà Nội" in response
            reason = "" if ok else "expected direct capital answer"
        if ok and is_rubbish(response):
            ok = False
            reason = "rubbish token detected"
        row = {
            "i": index,
            "prompt": prompt,
            "response": response,
            "expected_tool": expected_tool,
            "tool_name": tool_name,
            "parse_ok": parse_ok if "<tool_call>" in response else None,
            "ok": ok,
            "reason": reason,
        }
        rows.append(row)
        if not ok:
            failures.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({"pass": not failures, "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
