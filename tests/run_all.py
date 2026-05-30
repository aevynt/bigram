#!/usr/bin/env python
"""
run_all.py
==========
Chạy toàn bộ bộ kiểm thử của Bigram mà không cần cài pytest.

Cách dùng:
    python tests/run_all.py

Nếu mọi test đều pass, mã thoát = 0. Nếu có lỗi, mã thoát = 1.
"""

import os
import sys
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_model
import test_tokenizer
import test_upgrades
import test_data_training
import test_alignment
import test_tensor1_config
import test_tool_schema
import test_tool_registry
import test_tool_sft_dataset
import test_windows_tools
import test_rag
import test_server_import


def main():
    print("=" * 55)
    print("CHẠY TOÀN BỘ BỘ KIỂM THỬ BIGRAM")
    print("=" * 55 + "\n")

    suites = [
        ("Model", test_model.run_all),
        ("Tokenizer", test_tokenizer.run_all),
        ("Upgrades (VS-BPE/MLA/PonderNet/Mamba)", test_upgrades.run_all),
        ("Data & Training", test_data_training.run_all),
        ("Alignment (SFT/DPO/Calibration)", test_alignment.run_all),
        ("Tensor 1 Config", test_tensor1_config.run_all),
        ("Tool Schema", test_tool_schema.run_all),
        ("Tool Registry", test_tool_registry.run_all),
        ("Tool SFT Dataset", test_tool_sft_dataset.run_all),
        ("Windows Tools", test_windows_tools.run_all),
        ("RAG", test_rag.run_all),
        ("Server Import", test_server_import.run_all),
    ]

    failed = []
    for name, runner in suites:
        try:
            runner()
        except Exception:
            failed.append(name)
            print(f"!!! BỘ TEST '{name}' THẤT BẠI:")
            traceback.print_exc()
            print()

    print("=" * 55)
    if failed:
        print(f"KẾT QUẢ: {len(failed)} bộ test THẤT BẠI: {', '.join(failed)}")
        print("=" * 55)
        sys.exit(1)
    else:
        print("KẾT QUẢ: TẤT CẢ BỘ TEST ĐỀU PASS")
        print("=" * 55)
        sys.exit(0)


if __name__ == "__main__":
    main()
