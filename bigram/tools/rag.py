"""RAG search tool."""

from .base import BaseTool, ToolResult
from ..rag.search import lexical_search


class RagSearchTool(BaseTool):
    name = "rag.search"

    def __init__(self, default_index_path=None):
        self.default_index_path = default_index_path

    def run(self, args: dict) -> ToolResult:
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 5))
        index_path = args.get("index_path") or self.default_index_path
        if not index_path:
            return ToolResult(ok=False, output="", error="index_path is required")
        rows = lexical_search(index_path, query, top_k=top_k)
        lines = []
        for row in rows:
            lines.append(
                f"[{row['score']:.3f}] {row.get('title','')} | {row.get('source','')} | {row.get('url','')}\n{row.get('text','')}"
            )
        return ToolResult(ok=True, output="\n\n".join(lines), metadata={"results": rows})
