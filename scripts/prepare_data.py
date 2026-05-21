#!/usr/bin/env python
"""
prepare_data.py
===============
Bước 1 của pipeline: chuyển văn bản thô thành file nhị phân để train.

Cách dùng:
    python scripts/prepare_data.py \
        --tokenizer data/tokenizer.json \
        --input data/corpus.txt \
        --output-prefix data/train

Sẽ tạo ra: data/train.tok.bin và data/train.tone.bin

Thường ta chạy script này hai lần — một cho tập train, một cho tập validation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.tokenizer import BigramTokenizer
from bigram.data import prepare_corpus
from bigram.utils import format_number


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu cho Bigram")
    parser.add_argument("--tokenizer", required=True,
                        help="Đường dẫn file tokenizer.json")
    parser.add_argument("--input", required=True,
                        help="File .txt văn bản nguồn")
    parser.add_argument("--output-prefix", required=True,
                        help="Tiền tố output (sẽ thêm .tok.bin / .tone.bin)")
    parser.add_argument("--no-special", action="store_true",
                        help="Không bọc <bos>/<eos> cho từng dòng")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        print(f"LỖI: không tìm thấy tokenizer {args.tokenizer}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"LỖI: không tìm thấy file dữ liệu {args.input}")
        sys.exit(1)

    print(f"Nạp tokenizer từ {args.tokenizer}...")
    tokenizer = BigramTokenizer.load(args.tokenizer)

    print(f"Đang mã hóa {args.input}...")
    stats = prepare_corpus(
        args.input, tokenizer, args.output_prefix,
        add_special_per_line=not args.no_special,
    )

    print(f"Xong!")
    print(f"  Số dòng  : {stats['n_lines']}")
    print(f"  Số token : {format_number(stats['n_tokens'])}")
    print(f"  File token: {stats['tok_file']}")
    print(f"  File tone : {stats['tone_file']}")


if __name__ == "__main__":
    main()
