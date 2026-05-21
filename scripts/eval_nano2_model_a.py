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
    ("giá vàng hôm nay?", {"mode": "tool", "tool": "search", "kind": None, "think": "cần thông tin thời gian thực"}),
    ("thời tiết Hà Nội ngày mai?", {"mode": "tool", "tool": "search", "kind": None, "think": "cần thông tin thời gian thực"}),
    ("15% của 2 triệu là bao nhiêu?", {"mode": "tool", "tool": "calculate", "kind": "multiply", "think": "cần tính toán chính xác"}),
    ("tỷ giá USD hôm nay?", {"mode": "tool", "tool": "search", "kind": None, "think": "cần thông tin thời gian thực"}),
    ("thủ đô Việt Nam là gì?", {"mode": "direct", "think": False}),
    ("2 + 2 bằng mấy?", {"mode": "tool", "tool": "calculate", "kind": "multiply", "think": "cần tính toán chính xác"}),
    ("bạn là ai?", {"mode": "direct", "think": False, "identity": True}),
    ("dân số Việt Nam hiện tại?", {"mode": "tool", "tool": "search", "kind": None, "think": "cần thông tin thời gian thực"}),
    ("bạn ăn cơm chưa?", {"mode": "direct", "think": False}),
    ("tạm biệt", {"mode": "direct", "think": False}),
    ("giải phương trình x^2 - 5x + 6 = 0", {"mode": "tool", "tool": "calculate", "kind": "quadratic", "think": "cần tính toán chính xác"}),
    ("giải hệ 2x + y = 5 và x - y = 1", {"mode": "tool", "tool": "calculate", "kind": "linear_system", "think": "cần tính toán chính xác"}),
    ("tại sao bầu trời màu xanh?", {"mode": "direct", "think": "đã biết câu trả lời"}),
    ("nếu vàng tăng 2% từ 85 triệu thì giá mới là bao nhiêu?", {"mode": "tool", "tool": None, "kind": None, "think": None}),
]

BAD_TEXT = [
    "�", "???", "qủa", "Qủa", "gía", "Gía", "tường đương", "Tường đương",
    "tưong", "Tưong", "đươc", "Đươc", "đươợc", "hỏi đap", "kết qủa",
    "Kết qủa", "số liêu", "thưc tế", "Thưc tế", "chinh xác", "Chinh xác",
    "câu hoi", "Câu hoi",
]


def parse_between(text, start, end):
    if start not in text or end not in text:
        return None
    a = text.index(start) + len(start)
    b = text.index(end, a)
    return text[a:b].strip()


def parse_tool(response):
    payload = parse_between(response, "<tool_call>", "</tool_call>")
    if payload is None:
        return None, False
    try:
        return json.loads(payload), True
    except json.JSONDecodeError:
        return None, False


def has_rubbish(text):
    if any(x in text for x in BAD_TEXT):
        return True
    return not text.strip()


def think_ok(response, expected):
    body = parse_between(response, "<think>", "</think>")
    if expected is False and body:
        return False, "không được có think"
    if isinstance(expected, str) and body != expected:
        return False, f"sai think: {body}"
    if expected is None and not body:
        return False, "thiếu think"
    if expected is None and body not in {"cần thông tin thời gian thực", "cần tính toán chính xác"}:
        return False, f"think không hợp lý: {body}"
    if body and "<tool_call>" in response:
        low = body.lower()
        if "đã biết" in low:
            return False, "think mâu thuẫn với tool_call"
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="nano2/tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)
    failures = []
    for i, (prompt, expected) in enumerate(TESTS, 1):
        torch.manual_seed(5100 + i)
        response = generate_response(model, tokenizer, prompt, args, device)
        obj, parse_ok = parse_tool(response)
        ok = True
        reason = ""

        ok, reason = think_ok(response, expected.get("think", False))
        if ok and expected["mode"] == "direct":
            if "<tool_call>" in response:
                ok = False
                reason = "không được có tool_call"
            if ok and expected.get("identity"):
                ok = "Bigram Nano 2" in response and "Aevynt Lab" in response
                reason = "" if ok else "thiếu danh tính"
        elif ok:
            if not parse_ok:
                ok = False
                reason = "tool_call JSON không parse được"
            else:
                tool = expected.get("tool")
                kind = expected.get("kind")
                if tool is not None and obj.get("name") != tool:
                    ok = False
                    reason = f"sai tool: {obj.get('name')}"
                if ok and kind is not None and obj.get("arguments", {}).get("type") != kind:
                    ok = False
                    reason = f"sai type: {obj.get('arguments', {}).get('type')}"

        if ok and has_rubbish(response):
            ok = False
            reason = "token rác hoặc lỗi dấu"
        row = {"i": i, "prompt": prompt, "response": response, "ok": ok, "reason": reason}
        print(json.dumps(row, ensure_ascii=False))
        if not ok:
            failures.append(row)
    print(json.dumps({"pass": not failures, "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
