"""Tests for lexical RAG."""

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.rag import build_index, lexical_search


def test_build_and_search_tiny_index():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "party.md").write_text("Điều lệ Đảng quy định nguyên tắc tổ chức.", encoding="utf-8")
        out = root / "index.jsonl"
        stats = build_index(root, out)
        hits = lexical_search(out, "Điều lệ Đảng", top_k=1)
    assert stats["chunks"] == 1
    assert hits
    assert "Điều lệ Đảng" in hits[0]["text"]
    print("  [OK] test_build_and_search_tiny_index")


def run_all():
    print("Đang chạy test_rag...")
    test_build_and_search_tiny_index()
    print("test_rag: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
