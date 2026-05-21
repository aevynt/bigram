"""
Package `training` — vòng lặp huấn luyện và optimizer cho Bigram.

  - optim.py   : AdamW + cosine LR scheduler.
  - trainer.py : lớp Trainer điều phối training (pre-train, mid-train, SFT).
  - dpo.py     : Direct Preference Optimization (giai đoạn alignment).
"""

from .trainer import Trainer
from .optim import build_optimizer, get_lr, apply_lr
from .dpo import (
    dpo_loss, sequence_logprob, compute_dpo_loss_from_batch, DPOTrainer,
)

__all__ = [
    "Trainer", "build_optimizer", "get_lr", "apply_lr",
    "dpo_loss", "sequence_logprob", "compute_dpo_loss_from_batch",
    "DPOTrainer",
]
