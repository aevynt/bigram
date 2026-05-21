#!/usr/bin/env python
"""
train_dpo.py
============
Giai đoạn 3b của pipeline: Direct Preference Optimization (DPO).

DPO tinh chỉnh model theo SỞ THÍCH của con người. Cho mỗi câu hỏi, ta có một
câu trả lời "tốt hơn" (chosen) và một câu "kém hơn" (rejected). DPO dạy model
ưu tiên câu tốt hơn — mà không cần reward model riêng hay thuật toán RL phức
tạp như PPO. Vì sao chọn DPO: ổn định như học có giám sát, hợp với mục tiêu
"AI chính xác, đáng tin" của Bigram. Chi tiết: bigram/training/dpo.py.

QUAN TRỌNG: chạy DPO trên model ĐÃ QUA SFT, không phải model nền thô. Bản
thân model lúc bắt đầu DPO được sao chép làm "reference policy" đông cứng.

Định dạng dữ liệu — file .jsonl, mỗi dòng:
    {"prompt": "...", "chosen": "<trả lời tốt>", "rejected": "<trả lời kém>"}

Cách dùng:
    python scripts/train_dpo.py \
        --data data/preferences.jsonl \
        --tokenizer data/tokenizer.json \
        --init checkpoints_sft/ckpt_final.pt \
        --out-dir checkpoints_dpo \
        --beta 0.1
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer, BigramConfig
from bigram.config import ModelConfig
from bigram.data import PreferenceDataset
from bigram.training import DPOTrainer
from bigram.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="DPO cho Bigram")
    parser.add_argument("--data", required=True,
                        help="File .jsonl chứa prompt/chosen/rejected")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--init", required=True,
                        help="Checkpoint model đã qua SFT (BẮT BUỘC)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Hệ số DPO — kiểm soát mức bám sát model gốc (0.1-0.5)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out-dir", default="checkpoints_dpo")
    args = parser.parse_args()

    if not os.path.exists(args.init):
        print(f"LỖI: không tìm thấy checkpoint SFT {args.init}")
        print("DPO yêu cầu một model đã qua SFT để khởi tạo.")
        sys.exit(1)

    tokenizer = BigramTokenizer.load(args.tokenizer)

    # --- Nạp model đã qua SFT ---
    print(f"Nạp model SFT từ {args.init}...")
    ckpt = torch.load(args.init, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    config = BigramConfig()
    config.model = model_cfg
    model = BigramModel(model_cfg)
    model.load_state_dict(ckpt["model"])

    config.train.max_steps = args.max_steps
    config.train.out_dir = args.out_dir
    # DPO dùng learning rate rất nhỏ — chỉ tinh chỉnh nhẹ.
    config.train.learning_rate = 5e-6
    config.data.stage = "dpo"

    set_seed(config.train.seed)

    # --- Dataset ưu tiên ---
    dataset = PreferenceDataset(args.data, tokenizer,
                                block_size=config.model.max_seq_len)
    print(f"Số cặp ưu tiên: {len(dataset)}")

    # --- Train DPO ---
    trainer = DPOTrainer(model, config, dataset, beta=args.beta)
    trainer.train()


if __name__ == "__main__":
    main()
