#!/usr/bin/env python
import json
import random
from pathlib import Path


TRAIN = [
    {
        "prompt": "Giải thích đệ quy trong lập trình là gì?",
        "response": "Đệ quy là khi một hàm tự gọi lại chính nó để giải bài toán nhỏ hơn, và phải có điều kiện dừng.",
    },
    {
        "prompt": "Đệ quy là gì trong lập trình?",
        "response": "Đệ quy là kỹ thuật một hàm gọi lại chính nó. Nó cần điều kiện dừng để tránh lặp vô hạn.",
    },
    {
        "prompt": "Nói ngắn gọn về đệ quy",
        "response": "Đệ quy là cách giải bài toán bằng cách gọi lại cùng một hàm với phiên bản nhỏ hơn của bài toán.",
    },
    {
        "prompt": "Ví dụ đệ quy dễ hiểu",
        "response": "Ví dụ giai thừa: n! = n * (n-1)!, dừng ở 0! = 1. Đó là đệ quy.",
    },
]


VAL = [
    {
        "prompt": "Giải thích đệ quy trong lập trình là gì?",
        "response": "Đệ quy là khi một hàm tự gọi lại chính nó để xử lý bài toán nhỏ hơn, miễn là có điều kiện dừng.",
    }
]


def append_rows(path, rows):
    with Path(path).open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def count(path):
    return sum(1 for _ in Path(path).open("r", encoding="utf-8"))


def main():
    random.seed(43)
    train_rows = TRAIN * 20
    val_rows = VAL * 10
    random.shuffle(train_rows)
    append_rows("data/nano1_train.jsonl", train_rows)
    append_rows("data/nano1_val.jsonl", val_rows)
    print(json.dumps({
        "train_added": len(train_rows),
        "val_added": len(val_rows),
        "train_total": count("data/nano1_train.jsonl"),
        "val_total": count("data/nano1_val.jsonl"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
