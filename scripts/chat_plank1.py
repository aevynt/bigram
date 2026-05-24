#!/usr/bin/env python
"""
chat_plank1.py
==============
Giao diện hội thoại terminal cho Bigram Plank 1.

Cách dùng:
    python scripts/chat_plank1.py
    python scripts/chat_plank1.py --checkpoint plank1/sft/ckpt_final.pt
                                   --tokenizer plank1/tokenizer.json

Gõ 'quit' hoặc 'thoát' để kết thúc.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer
from bigram.config import ModelConfig


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = BigramModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, model_cfg


def generate_response(model, tokenizer, prompt: str,
                      max_new_tokens=60, recurrence=4,
                      temperature=0.4, top_k=15, device="cpu") -> str:
    """
    Sinh response cho một prompt, tách phần response ra khỏi prompt.
    """
    # Mã hóa prompt
    tok, tone = tokenizer.encode(prompt, add_special=False)
    bos = tokenizer.token_to_id("<bos>")
    tok = [bos] + tok
    if tone is None:
        tone = [0] * len(tok)
    else:
        tone = [0] + tone

    prompt_len = len(tok)  # độ dài gốc để cắt bỏ phần prompt khỏi output.

    token_ids = torch.tensor([tok], dtype=torch.long, device=device)
    tone_ids = torch.tensor([tone], dtype=torch.long, device=device)

    out_ids, out_tones, _ = model.generate(
        token_ids, tone_ids,
        max_new_tokens=max_new_tokens,
        num_recurrence=recurrence,
        temperature=temperature,
        top_k=top_k,
    )

    # Chỉ lấy phần mới sinh ra (bỏ phần prompt).
    new_ids = out_ids[0, prompt_len:].tolist()
    new_tones = out_tones[0, prompt_len:].tolist()

    # Bỏ token đặc biệt.
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    special = {tokenizer.token_to_id("<bos>"), eos, pad}

    # Cắt tại EOS nếu có.
    if eos in new_ids:
        cut = new_ids.index(eos)
        new_ids = new_ids[:cut]
        new_tones = new_tones[:cut]

    # Bỏ token đặc biệt còn sót.
    filtered_ids, filtered_tones = [], []
    for tid, tone in zip(new_ids, new_tones):
        if tid not in special:
            filtered_ids.append(tid)
            filtered_tones.append(tone)

    if not filtered_ids:
        return "..."

    return tokenizer.decode(filtered_ids, filtered_tones).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="plank1/sft/ckpt_final.pt")
    parser.add_argument("--tokenizer", default="plank1/tokenizer.json")
    parser.add_argument("--recurrence", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Đang nạp Bigram Plank 1 từ {args.checkpoint}...")
    model, model_cfg = load_model(args.checkpoint, device)
    tok_type = getattr(model_cfg, "tokenizer_type", "tonal")
    if tok_type == "bmssp":
        from bigram.tokenizer.bmssp import BMSSPTokenizer
        tokenizer = BMSSPTokenizer.load(args.tokenizer)
    else:
        tokenizer = BigramTokenizer.load(args.tokenizer)
    print("Sẵn sàng!\n")
    print("=" * 50)
    print("  Bigram Plank 1 — by Aevynt Lab")
    print("  Gõ 'quit' hoặc 'thoát' để thoát.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "thoát", "exit"):
            print("Bigram Plank 1: Tạm biệt! Hẹn gặp lại!")
            break

        response = generate_response(
            model, tokenizer, user_input,
            recurrence=args.recurrence,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"Bigram Plank 1: {response}\n")


if __name__ == "__main__":
    main()
