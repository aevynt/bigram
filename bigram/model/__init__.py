"""
Package `model` — chứa toàn bộ kiến trúc mạng nơ-ron của Bigram.

Các thành phần:
  - layers.py    : RMSNorm, RoPE, LayerScale.
  - attention.py : Grouped Query Attention.
  - ffn.py       : SwiGLU MLP và Mixture of Experts.
  - block.py     : một tầng transformer (sandwich norm).
  - bigram.py    : mô hình Bigram hoàn chỉnh (prelude/recurrent/coda).
  - loss.py      : các hàm mất mát.
"""

from .bigram import BigramModel, sample_recurrence
from .loss import compute_total_loss, language_modeling_loss, abstention_loss
from .layers import RMSNorm, RotaryEmbedding, LayerScale
from .attention import GroupedQueryAttention
from .ffn import SwiGLUMlp, MoEFFN
from .block import TransformerBlock
from .tooling import ToolHead, TOOL_REGISTRY_DEFAULT

__all__ = [
    "BigramModel",
    "sample_recurrence",
    "compute_total_loss",
    "language_modeling_loss",
    "abstention_loss",
    "RMSNorm",
    "RotaryEmbedding",
    "LayerScale",
    "GroupedQueryAttention",
    "SwiGLUMlp",
    "MoEFFN",
    "TransformerBlock",
    "ToolHead",
    "TOOL_REGISTRY_DEFAULT",
]
