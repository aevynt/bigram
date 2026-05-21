"""
Bigram — mô hình ngôn ngữ tiếng Việt với kiến trúc recurrent-depth.

Đây là package gốc. Import nhanh các thành phần chính:

    from bigram import BigramModel, BigramConfig, BigramTokenizer, Trainer

Cấu trúc package:
  - config.py        : toàn bộ siêu tham số.
  - model/           : kiến trúc mạng nơ-ron.
  - tokenizer/       : xử lý văn bản tiếng Việt (tách thanh điệu + BPE).
  - data/            : chuẩn bị và nạp dữ liệu.
  - training/        : vòng lặp huấn luyện.
  - utils/           : tiện ích phụ trợ.
"""

from .config import (
    BigramConfig, ModelConfig, TrainConfig, DataConfig,
    tiny_config, small_config,
)
from .model import BigramModel, compute_total_loss, sample_recurrence
from .tokenizer import BigramTokenizer
from .data import prepare_corpus, PackedDataset, JsonlSFTDataset
from .training import Trainer

__version__ = "0.1.0"

__all__ = [
    "BigramConfig", "ModelConfig", "TrainConfig", "DataConfig",
    "tiny_config", "small_config",
    "BigramModel", "compute_total_loss", "sample_recurrence",
    "BigramTokenizer",
    "prepare_corpus", "PackedDataset", "JsonlSFTDataset",
    "Trainer",
]
