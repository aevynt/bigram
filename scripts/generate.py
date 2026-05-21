#!/usr/bin/env python
"""
generate.py
===========
Bước 3 của pipeline: dùng model Bigram đã train để sinh văn bản (inference).

Cách dùng:
    python scripts/generate.py \
        --checkpoint checkpoints/ckpt_final.pt \
        --tokenizer data/tokenizer.json \
        --prompt "Thủ đô của Việt Nam là" \
        --recurrence 32 \
        --max-new-tokens 50

Tham số đáng chú ý:
  --recurrence          : số vòng "suy nghĩ". Câu khó -> đặt cao hơn (vd 48, 64).
                          Đây là cơ chế scale test-time compute của recurrent-depth.
  --abstention-threshold: nếu đặt (vd 0.5) và model thấy "không chắc" vượt ngưỡng,
                          nó sẽ DỪNG thay vì bịa — hiện thực hóa chống hallucination.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramConfig, BigramTokenizer
from bigram.config import ModelConfig
from bigram.utils import set_seed


def load_model_from_checkpoint(ckpt_path: str, device: str):
    """
    Nạp model từ checkpoint. Checkpoint chứa cả state_dict lẫn config model,
    nên ta dựng lại đúng kiến trúc rồi nạp trọng số.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # Dựng lại ModelConfig từ dict đã lưu trong checkpoint.
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = BigramModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, model_cfg


def main():
    parser = argparse.ArgumentParser(description="Sinh văn bản với Bigram")
    parser.add_argument("--checkpoint", required=True,
                        help="Đường dẫn checkpoint (.pt)")
    parser.add_argument("--tokenizer", required=True,
                        help="Đường dẫn tokenizer.json")
    parser.add_argument("--prompt", required=True,
                        help="Câu mở đầu để model viết tiếp")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Số token tối đa sẽ sinh")
    parser.add_argument("--recurrence", type=int, default=None,
                        help="Số vòng lặp recurrent (mặc định: theo config)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Độ 'sáng tạo'. Thấp -> an toàn, cao -> đa dạng")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Chỉ lấy mẫu trong top-k token có xác suất cao nhất")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold, for example 0.9")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Penalty for tokens already present in the context")
    parser.add_argument("--abstention-threshold", type=float, default=None,
                        help="Ngưỡng từ chối (0-1). Đặt để bật chống hallucination")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"LỖI: không tìm thấy checkpoint {args.checkpoint}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"Nạp model từ {args.checkpoint}...")
    model, model_cfg = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)

    # Mã hóa prompt.
    tok, tone = tokenizer.encode(args.prompt, add_special=False)
    # Thêm <bos> ở đầu để khớp cách model được train.
    bos = tokenizer.token_to_id("<bos>")
    tok = [bos] + tok
    tone = [0] + tone

    token_ids = torch.tensor([tok], dtype=torch.long, device=device)
    tone_ids = torch.tensor([tone], dtype=torch.long, device=device)

    print(f"Đang sinh (recurrence={args.recurrence or 'mặc định'})...\n")
    out_ids, out_tones, abstained = model.generate(
        token_ids, tone_ids,
        max_new_tokens=args.max_new_tokens,
        num_recurrence=args.recurrence,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        abstention_threshold=args.abstention_threshold,
    )

    # Giải mã kết quả. Dùng tone_ids để giữ đúng dấu thanh của phần prompt.
    result_ids = out_ids[0].tolist()
    result_tones = out_tones[0].tolist()
    text = tokenizer.decode(result_ids, result_tones)

    print("-" * 50)
    print(f"PROMPT : {args.prompt}")
    print(f"KẾT QUẢ: {text}")
    if abstained:
        print("\n[!] Model đã DỪNG vì không đủ tự tin (abstention).")
        print("    Đây là hành vi chống hallucination — model 'biết khi nào nên im lặng'.")
    print("-" * 50)


if __name__ == "__main__":
    main()
