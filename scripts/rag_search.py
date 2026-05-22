#!/usr/bin/env python
"""Search a Bigram lexical RAG JSONL index."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.rag import lexical_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for hit in lexical_search(args.index, args.query, args.top_k):
        print(f"[{hit['score']:.3f}] {hit.get('title','')} | {hit.get('source','')}")
        print(hit.get("text", ""))
        print()


if __name__ == "__main__":
    main()
