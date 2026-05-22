#!/usr/bin/env python
"""Tool supervised fine-tuning for Bigram Tensor 1."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramConfig, BigramModel, BigramTokenizer, Trainer, tensor1_config
from bigram.config import ModelConfig
from bigram.data import ToolSFTDataset
from bigram.utils import set_seed


def _load_config(args, tokenizer):
    if args.config:
        cfg = BigramConfig.load(args.config)
    elif args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        cfg = BigramConfig()
        cfg.model = ModelConfig(**ckpt["config"]["model"])
    else:
        cfg = tensor1_config()
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.model.use_tool_head = True
    cfg.model.use_verifier_head = True
    cfg.train.out_dir = args.out_dir
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    cfg.data.stage = "tool_sft"
    cfg.model.__post_init__()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Tool-SFT cho Bigram Tensor 1")
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--init", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="checkpoints/tensor1_tool_sft")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    tokenizer = BigramTokenizer.load(args.tokenizer)
    config = _load_config(args, tokenizer)
    model = BigramModel(config.model)
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Nạp init: missing={len(missing)} unexpected={len(unexpected)}")

    set_seed(config.train.seed)
    train_ds = ToolSFTDataset(args.data, tokenizer, config.model.max_seq_len)
    val_ds = None
    if args.val_data and os.path.exists(args.val_data):
        val_ds = ToolSFTDataset(args.val_data, tokenizer, config.model.max_seq_len)

    os.makedirs(config.train.out_dir, exist_ok=True)
    config.save(os.path.join(config.train.out_dir, "config.json"))
    trainer = Trainer(model, config, train_ds, val_dataset=val_ds)
    trainer.train()


if __name__ == "__main__":
    main()
