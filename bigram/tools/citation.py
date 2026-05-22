"""Naive citation verifier for early Tensor 1 experiments."""

import re

from .base import BaseTool, ToolResult


def _terms(text):
    return {t for t in re.split(r"\W+", (text or "").lower(), flags=re.UNICODE) if len(t) >= 3}


class CitationVerifyTool(BaseTool):
    name = "citation.verify"

    def run(self, args: dict) -> ToolResult:
        claims = args.get("claims", [])
        sources = args.get("sources", [])
        source_text = "\n".join(str(s.get("text", "")) for s in sources if isinstance(s, dict))
        source_terms = _terms(source_text)
        results = []
        for claim in claims:
            claim_text = str(claim)
            if claim_text and claim_text in source_text:
                status = "supported"
                score = 1.0
            else:
                terms = _terms(claim_text)
                overlap = len(terms & source_terms) / max(1, len(terms))
                if overlap >= 0.65:
                    status = "supported"
                elif overlap <= 0.2:
                    status = "unsupported"
                else:
                    status = "uncertain"
                score = overlap
            results.append({"claim": claim_text, "status": status, "score": score})
        output = "\n".join(f"{r['status']}: {r['claim']} ({r['score']:.2f})" for r in results)
        return ToolResult(ok=True, output=output, metadata={"results": results})
