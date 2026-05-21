#!/usr/bin/env python
"""
Chat UI rieng cho Bigram Nano.

Vi du:
    python scripts/chat_nano1.py
    python scripts/chat_nano1.py --once "Xin chao"
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer
from bigram.config import ModelConfig


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def enable_utf8_stdio():
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = BigramModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def generate_response(model, tokenizer, prompt, args, device):
    tok, tone = tokenizer.encode(prompt, add_special=False)
    bos = tokenizer.token_to_id("<bos>")
    tok = [bos] + tok
    tone = [0] + tone
    prompt_len = len(tok)

    token_ids = torch.tensor([tok], dtype=torch.long, device=device)
    tone_ids = torch.tensor([tone], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids, out_tones, _ = model.generate(
            token_ids,
            tone_ids,
            max_new_tokens=args.max_new_tokens,
            num_recurrence=args.recurrence,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    new_ids = out_ids[0, prompt_len:].tolist()
    new_tones = out_tones[0, prompt_len:].tolist()

    eos = tokenizer.token_to_id("<eos>")
    special = {
        tokenizer.token_to_id("<bos>"),
        tokenizer.token_to_id("<pad>"),
        eos,
    }
    if eos in new_ids:
        cut = new_ids.index(eos)
        new_ids = new_ids[:cut]
        new_tones = new_tones[:cut]

    clean_ids = []
    clean_tones = []
    for tid, tone_id in zip(new_ids, new_tones):
        if tid not in special:
            clean_ids.append(tid)
            clean_tones.append(tone_id)

    if not clean_ids:
        return "..."
    return tokenizer.decode(clean_ids, clean_tones).strip()


def typewrite(text, delay):
    for ch in text:
        print(ch, end="", flush=True)
        if ch == "\n":
            time.sleep(delay * 3)
        elif ch in ".!?":
            time.sleep(delay * 6)
        else:
            time.sleep(delay)
    print()


def banner(args, device):
    print()
    print(f"{CYAN}{BOLD}Bigram Nano{RESET} {DIM}by Aevynt Lab{RESET}")
    print(f"{DIM}checkpoint: {args.checkpoint}{RESET}")
    print(f"{DIM}device: {device} | temp={args.temperature} | top_k={args.top_k} | r={args.recurrence}{RESET}")
    print(f"{DIM}go 'quit', 'exit', hoac 'thoat' de dung{RESET}")
    print()


def run_once(model, tokenizer, args, device, prompt):
    print(f"{YELLOW}Bạn:{RESET} {prompt}")
    print(f"{GREEN}Nano:{RESET} ", end="", flush=True)
    response = generate_response(model, tokenizer, prompt, args, device)
    typewrite(response, args.type_delay)


def main():
    enable_utf8_stdio()
    parser = argparse.ArgumentParser(description="Chat voi Bigram Nano")
    parser.add_argument("--checkpoint", default="nano1/sft/ckpt_final.pt")
    parser.add_argument("--tokenizer", default="nano1/tokenizer.json")
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--type-delay", type=float, default=0.012,
                        help="Do tre moi ky tu khi hien thi response")
    parser.add_argument("--once", default=None,
                        help="Hoi mot cau roi thoat, tien cho test nhanh")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"{RED}Khong tim thay checkpoint: {args.checkpoint}{RESET}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"{RED}Khong tim thay tokenizer: {args.tokenizer}{RESET}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    model = load_model(args.checkpoint, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)

    banner(args, device)

    if args.once is not None:
        run_once(model, tokenizer, args, device, args.once.strip())
        return

    while True:
        try:
            prompt = input(f"{YELLOW}Bạn:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREEN}Nano:{RESET} ", end="", flush=True)
            typewrite("Chào nha, hẹn gặp lại.", args.type_delay)
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "thoat", "thoát"}:
            print(f"{GREEN}Nano:{RESET} ", end="", flush=True)
            typewrite("Chào nha, hẹn gặp lại.", args.type_delay)
            break

        print(f"{GREEN}Bigram Nano:{RESET} ", end="", flush=True)
        response = generate_response(model, tokenizer, prompt, args, device)
        typewrite(response, args.type_delay)
        print()


if __name__ == "__main__":
    main()
