"""Default tool registry for the API server."""

import os

from bigram.tools import (
    CalculatorTool,
    CitationVerifyTool,
    CodeSearchTool,
    MarkdownCheckTool,
    PythonRunTool,
    RagSearchTool,
    TerminalRunTool,
    ToolRegistry,
)


def create_default_tool_registry():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(MarkdownCheckTool())
    registry.register(PythonRunTool())
    registry.register(TerminalRunTool(safe_mode=True))
    registry.register(CodeSearchTool())
    registry.register(RagSearchTool(default_index_path=os.environ.get("BIGRAM_RAG_INDEX")))
    registry.register(CitationVerifyTool())
    return registry
