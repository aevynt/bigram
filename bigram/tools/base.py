"""Base classes for executable tools."""

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseTool:
    name: str

    def run(self, args: dict) -> ToolResult:
        raise NotImplementedError
