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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_model
import test_tokenizer
import test_data_training
import test_alignment


def main():
    print("=" * 55)
    print("CHẠY TOÀN BỘ BỘ KIỂM THỬ BIGRAM")
    print("=" * 55 + "\n")

    suites = [
        ("Model", test_model.run_all),
        ("Tokenizer", test_tokenizer.run_all),
        ("Data & Training", test_data_training.run_all),
        ("Alignment (SFT/DPO/Calibration)", test_alignment.run_all),
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
