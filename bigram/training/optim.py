"""
optim.py
========
Tạo optimizer và learning-rate scheduler cho Bigram.

Hai điểm đáng chú ý:
  1. Weight decay CHỌN LỌC: chỉ áp dụng cho ma trận trọng số (Linear, Embedding),
     KHÔNG áp dụng cho bias, RMSNorm, LayerScale. Đây là quy ước chuẩn — phạt L2
     lên các tham số 1 chiều thường gây hại.
  2. Lịch học cosine có warmup: LR tăng tuyến tính trong `warmup_steps` bước đầu,
     sau đó giảm theo hình cosine xuống `min_lr_ratio * lr`.
"""

import math
import torch


def build_optimizer(model, train_config):
    """
    Tạo AdamW với weight decay chọn lọc.

    Phân loại tham số:
      - decay   : tham số >= 2 chiều (ma trận Linear/Embedding) -> có weight decay.
      - no_decay: tham số 1 chiều (bias, gain của norm, gamma của LayerScale).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": train_config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
    )
    return optimizer


def get_lr(step: int, train_config) -> float:
    """
    Tính learning rate cho một bước cụ thể (cosine schedule + warmup).

    Giai đoạn 1 — warmup: step < warmup_steps
        lr tăng tuyến tính từ 0 -> learning_rate.
    Giai đoạn 2 — cosine decay: warmup_steps <= step <= max_steps
        lr giảm theo cosine từ learning_rate -> min_lr.
    Giai đoạn 3 — sau max_steps:
        lr giữ ở min_lr.
    """
    lr = train_config.learning_rate
    min_lr = lr * train_config.min_lr_ratio
    warmup = train_config.warmup_steps
    max_steps = train_config.max_steps

    # Warmup tuyến tính.
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)

    # Sau khi train xong -> giữ min_lr.
    if step >= max_steps:
        return min_lr

    # Cosine decay: tỉ lệ tiến trình từ 0 -> 1 trong vùng decay.
    progress = (step - warmup) / max(1, max_steps - warmup)
    # coeff đi từ 1 (đầu) xuống 0 (cuối) theo hình cosine.
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def apply_lr(optimizer, lr: float):
    """Gán learning rate mới cho tất cả param group của optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = lr
