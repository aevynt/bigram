#!/usr/bin/env python
"""Build a Bigram lexical RAG JSONL index."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.rag import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stats = build_index(args.input_dir, args.output)
    print(f"Đã build RAG index: {stats['chunks']} chunks -> {stats['output']}")


if __name__ == "__main__":
    main()
