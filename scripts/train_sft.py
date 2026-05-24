#!/usr/bin/env python
"""
train_sft.py
============
Giai đoạn 3a của pipeline: Supervised Fine-Tuning (SFT).

SFT biến model nền (base model, sau pre-training) thành một trợ lý biết TRẢ
LỜI. Ta train trên các cặp (câu hỏi -> câu trả lời tốt), và loss chỉ tính
trên phần câu trả lời — model học cách trả lời, không học lặp lại câu hỏi.

SFT là bước BẮT BUỘC trước DPO: nó đưa model vào đúng "vùng phân bố" của dữ
liệu hội thoại, để bước DPO sau đó học ưu tiên một cách ổn định.

Định dạng dữ liệu — file .jsonl, mỗi dòng:
    {"prompt": "<câu hỏi>", "response": "<câu trả lời tốt>"}

Cách dùng:
    python scripts/train_sft.py \
        --data data/sft.jsonl \
        --val-data data/val.jsonl \
        --tokenizer data/tokenizer.json \
        --init checkpoints/ckpt_final.pt \
        --out-dir checkpoints_sft
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer, Trainer
from bigram.config import ModelConfig, tiny_config, small_config
from bigram.data import JsonlSFTDataset
from bigram.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="SFT cho Bigram")
    parser.add_argument("--data", required=True,
                        help="File .jsonl chứa cặp prompt/response")
    parser.add_argument("--val-data", default=None,
                        help="File .jsonl validation để theo dõi val_loss (tùy chọn)")
    parser.add_argument("--tokenizer", required=True,
                        help="File tokenizer.json")
    parser.add_argument("--init", default=None,
                        help="Checkpoint model nền để khởi tạo (sau pre-training)")
    parser.add_argument("--preset", default="small",
                        choices=["tiny", "small"],
                        help="Preset config nếu không có --init")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="checkpoints_sft")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate cho SFT")
    args = parser.parse_args()

    # Tự động nhận diện loại tokenizer
    is_bmssp = False
    ckpt = None
    if args.init and os.path.exists(args.init):
        print(f"Khởi tạo model từ checkpoint nền: {args.init}")
        ckpt = torch.load(args.init, map_location="cpu")
        is_bmssp = ckpt.get("config", {}).get("model", {}).get("tokenizer_type", "tonal") == "bmssp"
    else:
        try:
            with open(args.tokenizer, "r", encoding="utf-8") as f:
                import json
                json.load(f)
            is_bmssp = False
        except Exception:
            is_bmssp = True

    if is_bmssp:
        from bigram.tokenizer.bmssp import BMSSPTokenizer
        tokenizer = BMSSPTokenizer.load(args.tokenizer)
        print("Đã nạp tokenizer dạng BMSSP")
    else:
        tokenizer = BigramTokenizer.load(args.tokenizer)
        print("Đã nạp tokenizer dạng Tonal BPE")

    # --- Dựng model: từ checkpoint nền, hoặc từ preset ---
    if ckpt is not None:
        model_cfg = ModelConfig(**ckpt["config"]["model"])
        from bigram import BigramConfig
        config = BigramConfig()
        config.model = model_cfg
        model = BigramModel(model_cfg)
        model.load_state_dict(ckpt["model"])
    else:
        print("Không có checkpoint nền — khởi tạo model mới từ preset.")
        print("LƯU Ý: SFT trên model chưa pre-train sẽ kém hiệu quả.")
        config = tiny_config() if args.preset == "tiny" else small_config()
        config.model.vocab_size = tokenizer.vocab_size
        model = BigramModel(config.model)

    # SFT thường dùng learning rate nhỏ hơn pre-training.
    config.train.learning_rate = args.lr
    config.train.max_steps = args.max_steps
    config.train.out_dir = args.out_dir
    config.train.log_interval = min(config.train.log_interval, 50)
    config.train.eval_interval = min(config.train.eval_interval, 50)
    config.train.save_interval = min(config.train.save_interval, 100)
    config.data.stage = "sft"

    set_seed(config.train.seed)

    # --- Dataset SFT ---
    dataset = JsonlSFTDataset(args.data, tokenizer,
                              block_size=config.model.max_seq_len)
    print(f"Số mẫu SFT train: {len(dataset)}")

    # --- Dataset validation (tùy chọn) ---
    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = JsonlSFTDataset(args.val_data, tokenizer,
                                      block_size=config.model.max_seq_len)
        print(f"Số mẫu SFT val: {len(val_dataset)}")

    trainer = Trainer(model, config, dataset, val_dataset=val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
