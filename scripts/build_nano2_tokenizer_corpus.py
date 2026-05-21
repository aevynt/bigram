#!/usr/bin/env python
import json
from pathlib import Path


TOOLS = ["search", "calculate", "get_weather", "translate", "summarize", "call_api"]
PARAMS = {
    "search": "query",
    "calculate": "expression",
    "get_weather": "location",
    "translate": "text",
    "summarize": "text",
    "call_api": "endpoint",
}


def json_lines():
    values = [
        "giá vàng",
        "tỷ giá USD",
        "thời tiết Hà Nội",
        "15% * 2000000",
        "hello world",
        "đoạn văn cần tóm tắt",
        "/products",
        "Không tìm thấy kết quả",
        "Giá vàng hôm nay là 85 triệu",
        "Hà Nội ngày mai có mưa nhẹ",
    ]
    lines = []
    for i in range(700):
        tool = TOOLS[i % len(TOOLS)]
        param = PARAMS[tool]
        value = values[i % len(values)]
        kind = i % 5
        if kind == 0:
            obj = {"name": tool, "arguments": {param: value}}
        elif kind == 1:
            obj = {"name": tool, "arguments": {param: value, "language": "tiếng Việt"}}
        elif kind == 2:
            obj = {"name": tool, "result": value}
        elif kind == 3:
            obj = {"name": tool, "error": "Không tìm thấy kết quả"}
        else:
            obj = {"name": tool, "arguments": {param: value}, "metadata": {"source": "nano2", "index": str(i)}}
        lines.append(json.dumps(obj, ensure_ascii=False))
    specials = ["{", "}", "[", "]", "\"", ":", ",", "{\"name\":", "\"arguments\":", "\"result\":", "\"error\":"]
    lines.extend(specials)
    return lines


def tag_lines():
    patterns = [
        "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"giá vàng\"}}\n</tool_call>",
        "<tool_call>\n{\"name\": \"calculate\", \"arguments\": {\"expression\": \"2000000 * 15 / 100\"}}\n</tool_call>",
        "<tool_result>\n{\"name\": \"search\", \"result\": \"Giá vàng hôm nay là 85 triệu\"}\n</tool_result>",
        "<tool_result>\n{\"name\": \"search\", \"error\": \"Không tìm thấy kết quả\"}\n</tool_result>",
        "<think>cần tìm kiếm thông tin mới nhất</think>",
        "<think>cần tính toán bằng công cụ calculate</think>",
        "[TOOLS]\n{\"name\": \"search\", \"description\": \"tìm kiếm thông tin\"}\n[/TOOLS]",
        "[TOOLS] search [/TOOLS]",
        "[TOOLS] search, calculate [/TOOLS]",
        "[NO TOOLS]",
    ]
    lines = []
    for i in range(200):
        lines.append(patterns[i % len(patterns)])
    return lines


def main():
    output = Path("data/nano2_tokenizer_corpus.txt")
    parts = []
    for path in [Path("data/plank1_corpus.txt"), Path("data/sample_corpus.txt")]:
        if path.exists():
            parts.extend(line.rstrip("\n") for line in path.open("r", encoding="utf-8", errors="replace"))
    parts.extend(json_lines())
    parts.extend(tag_lines())
    non_empty = [line for line in parts if line.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(non_empty) + "\n", encoding="utf-8")
    if len(non_empty) < 900:
        raise RuntimeError(f"Corpus quá ít dòng: {len(non_empty)}")
    print(json.dumps({"output": str(output), "non_empty_lines": len(non_empty)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
