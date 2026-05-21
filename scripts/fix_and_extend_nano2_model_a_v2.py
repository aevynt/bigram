#!/usr/bin/env python
import json
import random
import re
from pathlib import Path


REPLACEMENTS = [
    ("qủa", "quả"), ("Qủa", "Quả"),
    ("gía", "giá"), ("Gía", "Giá"),
    ("yều", "yêu"), ("Yều", "Yêu"),
    ("tưong", "tương"), ("Tưong", "Tương"),
    ("đươc", "được"), ("Đươc", "Được"),
    ("tường đương", "tương đương"),
    ("kết qủa", "kết quả"), ("Kết qủa", "Kết quả"),
]
BAD = [src for src, _ in REPLACEMENTS]
FORBIDDEN = ["—", "“", "”", "…"]
THINK_SEARCH = "<think>cần thông tin thời gian thực</think>"
THINK_CALC = "<think>cần tính toán chính xác</think>"
THINK_KNOWN = "<think>đã biết câu trả lời</think>"
FIXED_THINKS = {THINK_SEARCH, THINK_CALC, THINK_KNOWN}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def normalize_text(text):
    changed = False
    out = text
    for src, dst in REPLACEMENTS:
        if src in out:
            changed = True
            out = out.replace(src, dst)
    return out, changed


def between(text, start, end):
    if start not in text or end not in text:
        return None
    a = text.index(start) + len(start)
    b = text.index(end, a)
    return text[a:b].strip()


def parse_tool_call(response):
    payload = between(response, "<tool_call>", "</tool_call>")
    if payload is None:
        return None
    return json.loads(payload)


def canonicalize_think(response):
    if "<think>" not in response:
        return response, False
    body = between(response, "<think>", "</think>")
    if body is None:
        return response, False
    tool = parse_tool_call(response) if "<tool_call>" in response else None
    if tool and tool.get("name") == "search":
        new_think = THINK_SEARCH
    elif tool and tool.get("name") == "calculate":
        new_think = THINK_CALC
    elif not tool:
        new_think = THINK_KNOWN
    else:
        return response, False
    old = f"<think>{body}</think>"
    old_multiline = re.compile(r"<think>\s*.*?\s*</think>", re.S)
    updated = old_multiline.sub(new_think, response, count=1)
    return updated, updated != response or old != new_think


def has_bad_text(text):
    return any(x in text for x in BAD) or any(x in text for x in FORBIDDEN) or "�" in text


def validate_record(record):
    if not isinstance(record, dict):
        return False, "not_object"
    prompt = record.get("prompt")
    response = record.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return False, "missing_prompt_response"
    if has_bad_text(prompt) or has_bad_text(response):
        return False, "bad_text"
    if len(response) < 20:
        return False, "response_short"
    if "<tool_call>" in response:
        try:
            tool = parse_tool_call(response)
        except Exception:
            return False, "tool_json_invalid"
        if not isinstance(tool, dict) or not isinstance(tool.get("arguments"), dict):
            return False, "tool_json_invalid"
    if "<think>" in response:
        think_match = re.findall(r"<think>\s*(.*?)\s*</think>", response, flags=re.S)
        if len(think_match) != 1:
            return False, "think_count_invalid"
        fixed = f"<think>{think_match[0].strip()}</think>"
        if fixed not in FIXED_THINKS:
            return False, "think_not_fixed"
        has_tool = "<tool_call>" in response
        if has_tool and fixed == THINK_KNOWN:
            return False, "think_conflicts_tool"
        if not has_tool and fixed != THINK_KNOWN:
            return False, "think_conflicts_direct"
    return True, ""


def tool_call(name, arguments):
    return "<tool_call>\n" + dump({"name": name, "arguments": arguments}) + "\n</tool_call>"


def add(rows, prompt, response):
    record = {"prompt": prompt, "response": response}
    ok, reason = validate_record(record)
    if not ok:
        raise ValueError(f"{reason}: {record}")
    rows.append(record)


def fix_base():
    src = Path("data/nano2_a_sft.jsonl")
    dst = Path("data/nano2_a_sft_fixed.jsonl")
    total = normalized = dropped = 0
    reasons = {}
    examples = []
    output = []
    for line_no, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        before = line
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            reasons["json_parse"] = reasons.get("json_parse", 0) + 1
            continue
        prompt, p_changed = normalize_text(record.get("prompt", ""))
        response, r_changed = normalize_text(record.get("response", ""))
        record["prompt"] = prompt
        record["response"] = response
        try:
            response, t_changed = canonicalize_think(response)
            record["response"] = response
        except Exception:
            t_changed = False
        changed = p_changed or r_changed or t_changed
        if changed:
            normalized += 1
            if len(examples) < 5:
                examples.append({"line": line_no, "before": before, "after": dump(record)})
        ok, reason = validate_record(record)
        if not ok:
            dropped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        output.append(dump(record))
    dst.write_text("\n".join(output) + "\n", encoding="utf-8")
    return {"total": total, "normalized": normalized, "dropped": dropped, "reasons": reasons, "final": len(output), "examples": examples}


def build_extra():
    rows = []
    gold = [
        "giá vàng hôm nay?", "giá vàng hiện tại là bao nhiêu?", "giá vàng SJC hôm nay?",
        "giá vàng trong nước mới nhất?", "vàng hôm nay tăng hay giảm?", "giá vàng tuần này thế nào?",
        "giá vàng 9999 hôm nay?", "giá vàng nhẫn hôm nay?", "giá vàng thế giới hiện tại?",
        "giá vàng hôm nay so với hôm qua?",
    ]
    rates = [
        "tỷ giá USD hôm nay?", "tỷ giá EUR hôm nay?", "tỷ giá yên Nhật hiện tại?",
        "USD đổi sang VND bao nhiêu?", "tỷ giá đô la Mỹ mới nhất?", "tỷ giá ngân hàng hôm nay?",
        "tỷ giá ngoại tệ hiện tại?", "tỷ giá bảng Anh hôm nay?", "tỷ giá won Hàn hôm nay?",
        "tỷ giá nhân dân tệ hôm nay?",
    ]
    population = [
        "dân số Việt Nam hiện tại?", "dân số Hà Nội hiện tại?", "dân số Thành phố Hồ Chí Minh hiện tại?",
        "dân số thế giới hiện nay?", "dân số Nhật Bản hiện tại?", "dân số Mỹ hiện tại?",
        "dân số Đà Nẵng hiện tại?", "dân số Cần Thơ hiện tại?", "dân số châu Á hiện tại?",
        "dân số châu Âu hiện tại?",
    ]
    other_search = [
        "thời tiết Hà Nội ngày mai?", "thời tiết Đà Nẵng cuối tuần?", "tin tức mới nhất hôm nay?",
        "lịch thi đấu bóng đá tối nay?", "giá bitcoin hôm nay?", "giá xăng mới nhất?",
        "giá cổ phiếu VNM hiện tại?", "lãi suất ngân hàng hiện nay?", "lịch chiếu phim cuối tuần?",
        "tin công nghệ mới nhất?", "thời tiết Sài Gòn ngày mai?", "kết quả xổ số hôm nay?",
        "giá vé máy bay đi Đà Nẵng?", "lịch nghỉ lễ năm nay?", "giá dầu thô hiện tại?",
        "giá bạc hôm nay?", "tin chứng khoán mới nhất?", "lịch thi đấu Ngoại hạng Anh tối nay?",
        "nhiệt độ Hà Nội lúc này?", "giá nhà đất hiện nay?", "giá cà phê hôm nay?",
        "giá gạo xuất khẩu hiện tại?", "tin thời sự mới nhất?", "giá điện hiện hành?",
        "thời tiết Huế ngày mai?", "giá thép hôm nay?", "chỉ số VN-Index hiện tại?",
        "giá vé concert hôm nay?", "tin kinh tế mới nhất?", "dự báo mưa bão hôm nay?",
    ]
    for prompt in gold + rates + population + other_search:
        add(rows, prompt, THINK_SEARCH + "\n" + tool_call("search", {"query": prompt.rstrip("?")}))

    quadratic = [
        "giải phương trình x^2 - 5x + 6 = 0", "giải x^2 - 3x + 2 = 0",
        "tìm nghiệm x^2 - 4 = 0", "giải 2x^2 + 3x - 2 = 0",
        "giải x^2 + 2x + 1 = 0",
    ]
    multiply = [
        "15% của 2 triệu là bao nhiêu", "2 + 2 bằng mấy", "123 nhân 456 bằng bao nhiêu",
        "20% của 5 triệu là bao nhiêu", "85 triệu tăng 2% là bao nhiêu",
    ]
    systems = [
        "giải hệ 2x + y = 5 và x - y = 1", "giải hệ x + y = 10 và x - y = 4",
        "tìm x y biết x + 2y = 7 và 3x - y = 5", "giải hệ 3x + y = 8 và x + y = 4",
        "giải hệ 2x - y = 3 và x + y = 6",
    ]
    words = [
        "Một lớp 40 học sinh, 60% là nữ. Hỏi có bao nhiêu nữ?",
        "An có 5 kẹo, cho 2 rồi mua thêm 3, còn mấy cái?",
        "120 sản phẩm bán trong 3 ngày, trung bình mỗi ngày bao nhiêu?",
        "24 quả táo chia đều cho 6 bạn, mỗi bạn được mấy quả?",
        "Xe đi 60 km trong 2 giờ, vận tốc trung bình bao nhiêu?",
    ]
    for kind, prompts in [("quadratic", quadratic), ("multiply", multiply), ("linear_system", systems), ("word_problem", words)]:
        for i in range(20):
            prompt = prompts[i % len(prompts)]
            add(rows, prompt, THINK_CALC + "\n" + tool_call("calculate", {"type": kind, "expression": prompt}))

    known = [
        ("tại sao bầu trời màu xanh?", "Ánh sáng xanh bị tán xạ mạnh hơn trong khí quyển nên mắt ta thấy bầu trời có màu xanh."),
        ("đệ quy trong lập trình là gì?", "Đệ quy là cách một hàm tự gọi lại chính nó để giải bài toán nhỏ hơn."),
        ("vì sao nước biển mặn?", "Nước biển mặn vì muối khoáng từ đất đá theo sông suối chảy ra biển và tích tụ lâu dài."),
        ("nên học Python hay JavaScript trước?", "Nếu muốn học dữ liệu thì Python dễ bắt đầu hơn, còn làm web thì JavaScript hợp hơn."),
        ("công thức tính diện tích hình tròn là gì?", "Diện tích hình tròn bằng pi nhân bán kính bình phương, viết là S = pi r^2."),
    ]
    for i in range(30):
        prompt, answer = known[i % len(known)]
        add(rows, prompt, THINK_KNOWN + "\n" + answer)

    simple = [
        ("xin chào!", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("tạm biệt", "Tạm biệt, hẹn gặp lại bạn."),
        ("bạn là ai?", "Mình là Bigram Nano 2 do Aevynt Lab phát triển."),
        ("ai tạo ra bạn?", "Mình là Bigram Nano 2 của Aevynt Lab."),
        ("bạn ăn cơm chưa?", "Mình không ăn cơm, nhưng vẫn sẵn sàng trả lời bạn."),
        ("thủ đô Việt Nam là gì?", "Thủ đô Việt Nam là Hà Nội."),
    ]
    for i in range(30):
        prompt, answer = simple[i % len(simple)]
        add(rows, prompt, answer)

    if len(rows) != 200:
        raise RuntimeError(f"expected 200 extra rows, got {len(rows)}")
    Path("data/nano2_a_sft_extra.jsonl").write_text(
        "\n".join(dump(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def merge_split():
    rows = []
    for path in [Path("data/nano2_a_sft_fixed.jsonl"), Path("data/nano2_a_sft_extra.jsonl")]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            ok, reason = validate_record(record)
            if not ok:
                raise ValueError(f"{path}: {reason}: {record}")
            rows.append(record)
    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.9)
    train, val = rows[:split], rows[split:]
    Path("data/nano2_a_train_v2.jsonl").write_text("\n".join(dump(x) for x in train) + "\n", encoding="utf-8")
    Path("data/nano2_a_val_v2.jsonl").write_text("\n".join(dump(x) for x in val) + "\n", encoding="utf-8")
    return {"total": len(rows), "train": len(train), "val": len(val)}


def main():
    fixed = fix_base()
    extra = 0
    split = None
    if fixed["final"] >= 300:
        extra = build_extra()
        split = merge_split()
    print(json.dumps({"fixed": fixed, "extra": extra, "split": split}, ensure_ascii=False, indent=2))
    return 0 if fixed["final"] >= 300 else 2


if __name__ == "__main__":
    raise SystemExit(main())
