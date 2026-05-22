"""
Package `data` — chuẩn bị và nạp dữ liệu cho Bigram.

  - prepare.py : chuyển văn bản thô thành file nhị phân.
  - dataset.py : các lớp Dataset cho từng giai đoạn pipeline:
      * PackedDataset      — pre-training (giai đoạn 1, 2).
      * JsonlSFTDataset    — supervised fine-tuning (giai đoạn 3a).
      * PreferenceDataset  — DPO alignment (giai đoạn 3b).
      * CalibrationDataset — huấn luyện abstention head (giai đoạn 4).
"""

from .prepare import prepare_corpus
from .dataset import (
    PackedDataset, JsonlSFTDataset, PreferenceDataset, CalibrationDataset,
)
from .tool_sft import ToolSFTDataset

__all__ = [
    "prepare_corpus",
    "PackedDataset",
    "JsonlSFTDataset",
    "PreferenceDataset",
    "CalibrationDataset",
    "ToolSFTDataset",
]
