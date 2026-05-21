#!/usr/bin/env python
import argparse
import json
import re
import sys

import torch

sys.path.insert(0, ".")

from bigram import BigramTokenizer
from scripts.chat_nano1 import generate_response, load_model
from scripts.postprocess import TONE_FIXES, fix_output


TESTS = [
    (
        '<tool_result>\n{"name": "search", "query": "giá vàng hôm nay", "result": "Giá vàng hôm nay: 85 triệu đồng/lượng"}\n</tool_result>\nCâu hỏi gốc: giá vàng hôm nay?',
        ["vàng", "85"],
    ),
    (
        '<tool_result>\n{"name": "search", "query": "thời tiết Hà Nội ngày mai", "result": "Hà Nội ngày mai: mưa nhẹ, 24 độ C"}\n</tool_result>\nCâu hỏi gốc: thời tiết Hà Nội ngày mai?',
        ["Hà Nội", "mưa", "24"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "multiply", "result": 300000}\n</tool_result>\nCâu hỏi gốc: 15% của 2 triệu là bao nhiêu?',
        ["300"],
    ),
    (
        '<tool_result>\n{"name": "search", "query": "tỷ giá USD hôm nay", "result": "1 USD = 25,400 VND"}\n</tool_result>\nCâu hỏi gốc: tỷ giá USD hôm nay?',
        ["USD", "25"],
    ),
    (
        '<tool_result>\n{"name": "search", "query": "giá vé concert hôm nay", "error": "Không tìm thấy kết quả"}\n</tool_result>\nCâu hỏi gốc: giá vé concert hôm nay?',
        ["không"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "multiply", "result": 1250000}\n</tool_result>\nCâu hỏi gốc: chia đều 5 triệu cho 4 người?',
        ["1", "250"],
    ),
    (
        '<tool_result>\n{"name": "search", "query": "dân số Việt Nam hiện tại", "result": "Dân số Việt Nam 2024: khoảng 98 triệu người"}\n</tool_result>\nCâu hỏi gốc: dân số Việt Nam hiện tại?',
        ["Việt Nam", "98"],
    ),
    (
        '<tool_result>\n{"name": "search", "query": "giá bitcoin", "result": "Bitcoin hôm nay: $67,000"}\n</tool_result>\nCâu hỏi gốc: giá bitcoin?',
        ["Bitcoin", "67"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "quadratic", "result": {"x1": 2, "x2": 3}}\n</tool_result>\nCâu hỏi gốc: giải x^2 - 5x + 6 = 0',
        ["2", "3"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "multiply", "result": 56088}\n</tool_result>\nCâu hỏi gốc: 123 nhân 456 bằng bao nhiêu?',
        ["56088"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "linear_system", "result": {"x": 2, "y": 1}}\n</tool_result>\nCâu hỏi gốc: giải hệ 2x + y = 5 và x - y = 1',
        ["x", "2", "y", "1"],
    ),
    (
        '<tool_result>\n{"name": "calculate", "type": "word_problem", "result": {"result": "6", "steps": ["5-2=3", "3+3=6"]}}\n</tool_result>\nCâu hỏi gốc: An có 5 kẹo cho 2 mua thêm 3 còn mấy?',
        ["6"],
    ),
]

BAD = list(TONE_FIXES.keys()) + ["�", "???", "<tool_call>", "<tool_result>", "</tool_result>", "{", "}", "[", "]"]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip())


def has_rubbish(text):
    if any(x in text for x in BAD):
        return True
    if not text.strip() or len(text.strip()) < 8:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="nano2/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)
    failures = []
    for i, (prompt, required) in enumerate(TESTS, 1):
        torch.manual_seed(4200 + i)
        raw = normalize(generate_response(model, tokenizer, prompt, args, device))
        response = fix_output(raw)
        lower = response.lower()
        ok = True
        reason = ""
        for token in required:
            if token.lower() not in lower:
                ok = False
                reason = f"thiếu nội dung bắt buộc: {token}"
                break
        if ok and has_rubbish(response):
            ok = False
            reason = "có tag/json thô, lỗi dấu hoặc token rác"
        row = {"i": i, "prompt": prompt, "raw": raw, "response": response, "ok": ok, "reason": reason}
        print(json.dumps(row, ensure_ascii=False))
        if not ok:
            failures.append(row)
    print(json.dumps({"pass": not failures, "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
