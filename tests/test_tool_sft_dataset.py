"""Tests for ToolSFTDataset."""

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.data import ToolSFTDataset
from bigram.tokenizer import BigramTokenizer


def test_tool_sft_dataset_outputs_targets():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        corpus = root / "corpus.txt"
        corpus.write_text("user assistant tool rag search final answer Điều lệ Đảng\n" * 20, encoding="utf-8")
        tok = BigramTokenizer.train([str(corpus)], vocab_size=256, min_frequency=1)
        data = root / "tool.jsonl"
        row = {
            "messages": [
                {"role": "user", "content": "Tra cứu Điều lệ Đảng"},
                {"role": "assistant", "content": "<tool_call>{\"tool\":\"rag.search\",\"args\":{\"query\":\"Điều lệ Đảng\"}}</tool_call>"},
                {"role": "tool", "content": "<tool_result>nguồn</tool_result>"},
                {"role": "assistant", "content": "Câu trả lời có nguồn."},
            ]
        }
        data.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        ds = ToolSFTDataset(str(data), tok, block_size=96)
        sample = ds[0]
    assert sample["token_ids"].shape[0] == 96
    assert (sample["targets"] != -100).any()
    assert (sample["tool_router_targets"] == 1).any()
    assert (sample["tool_name_targets"] == 2).any()
    print("  [OK] test_tool_sft_dataset_outputs_targets")


def run_all():
    print("Đang chạy test_tool_sft_dataset...")
    test_tool_sft_dataset_outputs_targets()
    print("test_tool_sft_dataset: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
