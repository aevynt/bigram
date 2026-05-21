#!/usr/bin/env python
import json
import sys

import torch

sys.path.insert(0, ".")

from bigram import BigramTokenizer
from scripts.pipeline_nano2 import generate, load_model, parse_model_a_output
from scripts.postprocess import TONE_FIXES, fix_output


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

BAD = list(TONE_FIXES.keys()) + ["�", "???"]


def has_bad_text(text):
    return any(item in text for item in BAD) or not text.strip()


def check(parsed, fixed, expected):
    think = parsed["think"]
    action = parsed["tool_call"]

    if expected["think"] is False and think is not None:
        return False, f"không được có think: {think}"
    if isinstance(expected["think"], str) and think != expected["think"]:
        return False, f"sai think: {think}"
    if expected["think"] is None and think not in {"cần thông tin thời gian thực", "cần tính toán chính xác"}:
        return False, f"think không hợp lý: {think}"

    if expected["mode"] == "direct":
        if action is not None:
            return False, "không được có tool_call"
        if expected.get("identity") and ("Bigram Nano 2" not in fixed or "Aevynt Lab" not in fixed):
            return False, "thiếu danh tính"
    else:
        if not isinstance(action, dict):
            return False, "tool_call không parse được"
        if expected.get("tool") and action.get("name") != expected["tool"]:
            return False, f"sai tool: {action.get('name')}"
        kind = expected.get("kind")
        if kind and action.get("arguments", {}).get("type") != kind:
            return False, f"sai type: {action.get('arguments', {}).get('type')}"

    if has_bad_text(fixed):
        return False, "còn lỗi dấu hoặc token rác sau fix"
    return True, ""


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("nano2/model_a/sft/ckpt_final.pt", device)
    tokenizer = BigramTokenizer.load("nano2/tokenizer.json")
    failures = []
    for i, (prompt, expected) in enumerate(TESTS, 1):
        torch.manual_seed(5100 + i)
        raw = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=140,
            temperature=0.2,
            top_k=3,
            recurrence=4,
            device=device,
        )
        fixed = fix_output(raw)
        parsed = parse_model_a_output(raw)
        ok, reason = check(parsed, fixed, expected)
        row = {
            "i": i,
            "prompt": prompt,
            "raw": raw,
            "fixed": fixed,
            "think": parsed["think"],
            "action": parsed["tool_call"] or "direct",
            "ok": ok,
            "reason": reason,
        }
        print(json.dumps(row, ensure_ascii=False))
        if not ok:
            failures.append(row)
    print(json.dumps({"pass": not failures, "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
