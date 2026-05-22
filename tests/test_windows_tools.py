"""Tests for Windows-safe local tools."""

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.tools import CalculatorTool, CodeSearchTool, MarkdownCheckTool, TerminalRunTool


def test_calculator_tool():
    result = CalculatorTool().run({"expression": "(10 - 2) / 4"})
    assert result.ok
    assert result.output == "2.0"
    print("  [OK] test_calculator_tool")


def test_markdown_check_tool():
    result = MarkdownCheckTool().run({"text": "#Bad\n```python\nprint(1)"})
    assert not result.ok
    assert "unclosed code fence" in result.output
    print("  [OK] test_markdown_check_tool")


def test_code_search_tool_temp_dir():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.txt"
        path.write_text("hello Tensor 1\n", encoding="utf-8")
        result = CodeSearchTool().run({"query": "Tensor 1", "root": d, "max_results": 5})
    assert result.ok
    assert "Tensor 1" in result.output
    print("  [OK] test_code_search_tool_temp_dir")


def test_terminal_blocklist():
    result = TerminalRunTool().run({"command": "Remove-Item -Recurse -Force C:\\"})
    assert not result.ok
    assert "blocked" in result.error
    print("  [OK] test_terminal_blocklist")


def run_all():
    print("Đang chạy test_windows_tools...")
    test_calculator_tool()
    test_markdown_check_tool()
    test_code_search_tool_temp_dir()
    test_terminal_blocklist()
    print("test_windows_tools: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
