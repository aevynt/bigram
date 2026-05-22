"""Model/tokenizer/checkpoint loading helpers for Windows server deploy."""

import json
import os
from pathlib import Path

import torch

from bigram import BigramConfig, BigramModel, BigramTokenizer, tensor1_config
from bigram.config import ModelConfig


def resolve_device():
    requested = os.environ.get("BIGRAM_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device):
    requested = (os.environ.get("BIGRAM_DTYPE") or "").lower()
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_config(path):
    if not path or not Path(path).exists():
        return tensor1_config()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "model" in raw and "train" in raw and "data" in raw:
        return BigramConfig.load(path)
    cfg = tensor1_config()
    if "model" in raw:
        for key, value in raw["model"].items():
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
        cfg.model.__post_init__()
    if "train" in raw:
        for key, value in raw["train"].items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)
    return cfg


def load_tokenizer(path):
    if not path or not Path(path).exists():
        return None
    return BigramTokenizer.load(path)


def load_checkpoint(path, model):
    if not path or not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            return False, "safetensors is not installed"
        state = load_file(path)
        model.load_state_dict(state, strict=False)
        return True, None
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return True, None


def create_model_bundle():
    device = resolve_device()
    dtype = resolve_dtype(device)
    cfg = load_config(os.environ.get("BIGRAM_CONFIG"))
    tokenizer = load_tokenizer(os.environ.get("BIGRAM_TOKENIZER"))
    if tokenizer is not None and tokenizer.vocab_size != cfg.model.vocab_size:
        cfg.model.vocab_size = tokenizer.vocab_size
    checkpoint = os.environ.get("BIGRAM_CHECKPOINT")
    if tokenizer is None or not checkpoint or not Path(checkpoint).exists():
        return {
            "model": None,
            "tokenizer": tokenizer,
            "config": cfg,
            "device": device,
            "dtype": dtype,
            "error": "model not loaded: tokenizer or checkpoint missing",
        }
    model = BigramModel(cfg.model)
    ok, error = load_checkpoint(checkpoint, model)
    if not ok:
        return {"model": None, "tokenizer": tokenizer, "config": cfg, "device": device, "dtype": dtype, "error": error}
    model.to(device=device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "config": cfg, "device": device, "dtype": dtype, "error": None}
