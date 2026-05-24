#!/usr/bin/env python
"""
train.py
========
Bước 2 của pipeline: huấn luyện model Bigram.

Cách dùng cơ bản (dùng config mặc định 'small'):
    python scripts/train.py \
        --train-data data/train \
        --val-data data/val \
        --tokenizer data/tokenizer.json

Dùng config tùy chỉnh từ file JSON:
    python scripts/train.py --config configs/my_config.json ...

Train tiếp từ một checkpoint:
    python scripts/train.py --resume checkpoints/ckpt_step1000.pt ...

Ghi chú về `--preset`:
  - tiny  : model siêu nhỏ, chạy được trên CPU, dùng để thử nghiệm.
  - small : 'Bigram-0.3B', proof-of-concept thực sự (cần GPU).
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import (
    BigramConfig, BigramModel, BigramTokenizer, Trainer, PackedDataset,
)
from bigram.config import tiny_config, small_config
from bigram.utils import set_seed, count_parameters, format_number, \
    estimate_effective_depth


def main():
    parser = argparse.ArgumentParser(description="Train model Bigram")
    parser.add_argument("--train-data", required=True,
                        help="Tiền tố file dữ liệu train (vd: data/train)")
    parser.add_argument("--val-data", default=None,
                        help="Tiền tố file dữ liệu validation (tùy chọn)")
    parser.add_argument("--tokenizer", default=None,
                        help="File tokenizer (để lấy đúng vocab_size)")
    parser.add_argument("--config", default=None,
                        help="File config JSON (ghi đè --preset)")
    parser.add_argument("--preset", default="small",
                        choices=["tiny", "small"],
                        help="Cấu hình dựng sẵn nếu không dùng --config")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint để train tiếp")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Ghi đè số bước train")
    parser.add_argument("--out-dir", default=None,
                        help="Ghi đè thư mục lưu checkpoint")
    args = parser.parse_args()

    # --- Dựng config ---
    if args.config:
        config = BigramConfig.load(args.config)
        print(f"Đã nạp config từ {args.config}")
    else:
        config = tiny_config() if args.preset == "tiny" else small_config()
        print(f"Dùng config preset: {args.preset}")

    # Ghi đè vài tham số nếu người dùng chỉ định.
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.out_dir is not None:
        config.train.out_dir = args.out_dir

    # --- Đồng bộ vocab_size với tokenizer (nếu cung cấp) ---
    if args.tokenizer and os.path.exists(args.tokenizer):
        tok_type = getattr(config.model, "tokenizer_type", "tonal")
        if tok_type == "bmssp":
            from bigram.tokenizer.bmssp import BMSSPTokenizer
            tok = BMSSPTokenizer.load(args.tokenizer)
        else:
            tok = BigramTokenizer.load(args.tokenizer)
        if tok.vocab_size != config.model.vocab_size:
            print(f"Điều chỉnh vocab_size: {config.model.vocab_size} "
                  f"-> {tok.vocab_size} (theo tokenizer)")
            config.model.vocab_size = tok.vocab_size

    set_seed(config.train.seed)

    # --- Nạp dữ liệu ---
    block = config.model.max_seq_len
    train_ds = PackedDataset(
        args.train_data + ".tok.bin",
        args.train_data + ".tone.bin",
        block_size=block,
    )
    val_ds = None
    if args.val_data:
        val_ds = PackedDataset(
            args.val_data + ".tok.bin",
            args.val_data + ".tone.bin",
            block_size=block,
        )

    # --- Khởi tạo model ---
    model = BigramModel(config.model)

    # In thông tin model.
    print("\n" + "=" * 50)
    print("THÔNG TIN MODEL BIGRAM")
    print("=" * 50)
    groups = count_parameters(model)
    for k, v in groups.items():
        print(f"  {k:12s}: {format_number(v)}")
    print(f"  Độ sâu hiệu dụng (r={int(config.model.mean_recurrence)}): "
          f"{estimate_effective_depth(config.model)} layer")
    print(f"  Số block train: {len(train_ds)}")
    print("=" * 50 + "\n")

    # --- Train ---
    trainer = Trainer(model, config, train_ds, val_ds)
    if args.resume:
        print(f"Train tiếp từ checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Lưu lại config để tái lập.
    os.makedirs(config.train.out_dir, exist_ok=True)
    config.save(os.path.join(config.train.out_dir, "config.json"))

    trainer.train()


if __name__ == "__main__":
    main()
