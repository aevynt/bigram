"""Simple Vietnamese-friendly lexical search."""

import json
import re


def _tokens(text):
    return [t for t in re.split(r"[^\wÀ-ỹĐđ]+", (text or "").lower(), flags=re.UNICODE) if t]


def _score(row, query):
    q = _tokens(query)
    if not q:
        return 0.0
    text = row.get("text", "")
    hay = " ".join([row.get("title", ""), text]).lower()
    terms = set(_tokens(hay))
    overlap = sum(1 for term in q if term in terms)
    phrase_bonus = 2.0 if query.lower() in hay else 0.0
    title_bonus = 0.5 if any(term in row.get("title", "").lower() for term in q) else 0.0
    return overlap / len(q) + phrase_bonus + title_bonus


def lexical_search(index_path, query, top_k=5):
    hits = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score = _score(row, query)
            if score > 0:
                row = dict(row)
                row["score"] = score
                hits.append(row)
    hits.sort(key=lambda r: r["score"], reverse=True)
    return hits[:top_k]
