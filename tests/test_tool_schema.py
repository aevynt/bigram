"""Tests for agent tool schema."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.agent.schema import parse_tool_call, render_tool_call


def test_render_parse_tool_call():
    text = render_tool_call("rag.search", {"query": "Điều lệ Đảng"})
    call = parse_tool_call(text)
    assert call is not None
    assert call.tool == "rag.search"
    assert call.args["query"] == "Điều lệ Đảng"
    print("  [OK] test_render_parse_tool_call")


def test_invalid_json_returns_none():
    assert parse_tool_call("<tool_call>{bad</tool_call>") is None
    print("  [OK] test_invalid_json_returns_none")


def run_all():
    print("Đang chạy test_tool_schema...")
    test_render_parse_tool_call()
    test_invalid_json_returns_none()
    print("test_tool_schema: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
