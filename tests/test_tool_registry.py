"""Tests for tool registry."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.tools import CalculatorTool, ToolRegistry


def test_register_calculator():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.run("calculator", {"expression": "2+3*4"})
    assert result.ok
    assert result.output == "14"
    print("  [OK] test_register_calculator")


def test_unknown_tool_returns_error():
    registry = ToolRegistry()
    result = registry.run("missing.tool", {})
    assert not result.ok
    assert "unknown tool" in result.error
    print("  [OK] test_unknown_tool_returns_error")


def run_all():
    print("Đang chạy test_tool_registry...")
    test_register_calculator()
    test_unknown_tool_returns_error()
    print("test_tool_registry: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
