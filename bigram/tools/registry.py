"""Simple in-process tool registry."""

from .base import ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool
        return tool

    def run(self, name: str, args: dict | None = None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output="", error=f"unknown tool: {name}")
        try:
            return tool.run(args or {})
        except Exception as exc:
            return ToolResult(ok=False, output="", error=str(exc))

    def list_tools(self):
        return sorted(self._tools)
