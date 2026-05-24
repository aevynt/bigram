#!/usr/bin/env python
"""
train_tokenizer.py
==================
Bước 0 của pipeline: huấn luyện BigramTokenizer từ dữ liệu văn bản tiếng Việt.

Cách dùng:
    python scripts/train_tokenizer.py \
        --input data/corpus.txt \
        --output data/tokenizer.json \
        --vocab-size 32000

Có thể truyền nhiều file:
    python scripts/train_tokenizer.py --input a.txt b.txt c.txt --output tok.json

LƯU Ý: tokenizer phải được train MỘT LẦN và cố định. Mọi bước sau (chuẩn bị
dữ liệu, train model) đều phụ thuộc vào nó. Đổi tokenizer = phải làm lại từ đầu.
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cho phép import package `bigram` khi chạy script trực tiếp.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.tokenizer import BigramTokenizer
from bigram.tokenizer.bmssp import BMSSPTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BigramTokenizer")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Một hoặc nhiều file .txt văn bản tiếng Việt")
    parser.add_argument("--output", default="data/tokenizer.json",
                        help="Đường dẫn lưu tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Kích thước từ điển mong muốn")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Tần suất tối thiểu để gộp một cặp BPE")
    parser.add_argument("--tokenizer-type", default="tonal", choices=["tonal", "bmssp"],
                        help="Loại tokenizer: tonal hoặc bmssp")
    args = parser.parse_args()

    # Kiểm tra file đầu vào tồn tại.
    for path in args.input:
        if not os.path.exists(path):
            print(f"LỖI: không tìm thấy file {path}")
            sys.exit(1)

    print(f"Đang train tokenizer ({args.tokenizer_type}) từ {len(args.input)} file...")
    print(f"  Vocab size mục tiêu: {args.vocab_size}")

    if args.tokenizer_type == "bmssp":
        tokenizer = BMSSPTokenizer.train(
            args.input,
            vocab_size=args.vocab_size,
        )
    else:
        tokenizer = BigramTokenizer.train(
            args.input,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )

    # Tạo thư mục output nếu chưa có.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tokenizer.save(args.output)

    print(f"Xong! Vocab thực tế: {tokenizer.vocab_size}")
    print(f"Đã lưu tokenizer tại: {args.output}")

    # Thử nghiệm nhanh.
    demo = "Tiếng Việt là ngôn ngữ giàu thanh điệu."
    tok, tone = tokenizer.encode(demo)
    decoded = tokenizer.decode(tok, tone)
    print(f"\nKiểm tra nhanh:")
    print(f"  Gốc    : {demo}")
    print(f"  Mã hóa : {len(tok)} token")
    print(f"  Giải mã: {decoded}")


if __name__ == "__main__":
    main()
