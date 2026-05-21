#!/usr/bin/env python
import json
from pathlib import Path


IDENTITY_TERMS = [
    "là ai",
    "tên gì",
    "tạo ra",
    "ai tạo",
    "do ai tạo",
    "AI",
    "model",
]
FINANCE_TERMS = ["giá", "tỷ giá", "cổ phiếu", "lãi suất"]
STATE_TERMS = [
    "khỏe",
    "mệt",
    "đang làm gì",
    "vui",
    "bận",
    "ổn",
    "thế nào",
    "buồn",
    "đang nghĩ",
    "có sao",
    "thích gì",
    "ngủ",
    "cảm xúc",
    "đang ở đâu",
    "ăn cơm",
]

STATE_POOL = [
    ("bạn có khỏe không?", "Mình ổn nha, sẵn sàng hỗ trợ bạn."),
    ("bạn mệt chưa?", "Chưa mệt nha, mình vẫn trả lời được."),
    ("bạn đang làm gì?", "Mình đang đọc câu hỏi của bạn và chuẩn bị trả lời."),
    ("bạn vui không?", "Mình vui theo kiểu trợ lý, vì đang được trò chuyện."),
    ("bạn bận không?", "Không bận đâu, bạn cần gì cứ hỏi."),
    ("bạn ổn không?", "Ổn nha, mình vẫn hoạt động bình thường."),
    ("bạn thế nào?", "Mình ổn, đang sẵn sàng giúp bạn."),
    ("bạn có buồn không?", "Mình không buồn kiểu người, nhưng vẫn trả lời thân thiện."),
    ("bạn đang nghĩ gì?", "Mình đang tập trung vào câu hỏi của bạn."),
    ("bạn có sao không?", "Không sao nha, mình vẫn ở đây."),
    ("bạn thích gì?", "Mình thích dữ liệu sạch, câu hỏi rõ, và trả lời gọn."),
    ("bạn có ngủ không?", "Mình không ngủ như người, chỉ im lặng khi không có prompt."),
    ("bạn có cảm xúc không?", "Mình không có cảm xúc thật, nhưng vẫn trả lời thân thiện."),
    ("bạn có mệt không?", "Không mệt nha, mình vẫn xử lý được."),
    ("bạn đang ở đâu?", "Mình đang ở trong phiên chat này."),
    ("bạn ăn cơm chưa?", "Mình không ăn cơm, nhưng vẫn đủ năng lượng token để trả lời bạn."),
]

FORBIDDEN = {"\u2014", "\u201c", "\u201d", "\u2018", "\u2019", "\u2026"}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_any(prompt, terms):
    lower = prompt.lower()
    return any(term.lower() in lower for term in terms)


def is_identity(row):
    return has_any(row["prompt"], IDENTITY_TERMS)


def is_finance(row):
    return has_any(row["prompt"], FINANCE_TERMS)


def is_state(row):
    prompt = row["prompt"].lower().strip()
    return prompt.startswith("bạn") and any(term in prompt for term in STATE_TERMS)


def validate_new(row):
    text = row["prompt"] + row["response"]
    if any(ch in text for ch in FORBIDDEN):
        raise RuntimeError(f"Forbidden char in {row}")
    if len(row["response"]) > 100:
        raise RuntimeError(f"Response too long: {row}")
    lower = row["response"].lower()
    if "tài chính" in lower or "dữ liệu thời gian thực" in lower:
        raise RuntimeError(f"Bad state response: {row}")


def counts(rows):
    return {
        "identity": sum(is_identity(row) for row in rows),
        "finance": sum(is_finance(row) for row in rows),
        "state": sum(is_state(row) for row in rows),
    }


def main():
    rows = load_jsonl("data/nano1_train.jsonl")
    before = counts(rows)
    need = max(0, before["finance"] - before["state"])
    additions = []
    for i in range(need):
        prompt, response = STATE_POOL[i % len(STATE_POOL)]
        row = {"prompt": prompt, "response": response}
        validate_new(row)
        additions.append(row)
    balanced = rows + additions
    write_jsonl("data/nano1_train_balanced.jsonl", balanced)
    after = counts(balanced)
    print(json.dumps({
        "before": before,
        "added_state": len(additions),
        "after": after,
        "total_before": len(rows),
        "total_after": len(balanced),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
