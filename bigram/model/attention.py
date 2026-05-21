"""
attention.py
============
Grouped Query Attention (GQA) — cơ chế chú ý của Bigram.

GQA là điểm trung gian giữa:
  - Multi-Head Attention (MHA): mỗi query head có key/value head riêng. Tốn KV-cache.
  - Multi-Query Attention (MQA): mọi query head dùng chung 1 key/value head. Nhanh nhưng yếu.

GQA: chia query head thành các NHÓM, mỗi nhóm dùng chung một key/value head.
Ví dụ n_heads=8, n_kv_heads=2 -> mỗi kv head phục vụ 4 query head.
=> Giảm mạnh bộ nhớ KV-cache lúc suy luận (quan trọng cho văn bản dài) mà
   chất lượng gần như không giảm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryEmbedding, apply_rotary


class GroupedQueryAttention(nn.Module):
    """Self-attention nhân quả (causal) kiểu Grouped Query Attention + RoPE."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        # Mỗi kv head được "nhân bản" cho bao nhiêu query head.
        self.n_rep = self.n_heads // self.n_kv_heads

        # Phép chiếu tuyến tính cho Q, K, V.
        # Q chiếu ra n_heads * head_dim; K,V chỉ chiếu ra n_kv_heads * head_dim (ít hơn).
        self.q_proj = nn.Linear(config.hidden_size,
                                self.n_heads * self.head_dim,
                                bias=config.attn_qk_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.n_kv_heads * self.head_dim,
                                bias=config.attn_qk_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.n_kv_heads * self.head_dim,
                                bias=False)  # V không cần bias (theo Huginn).
        # Phép chiếu gộp các head lại về hidden_size.
        self.o_proj = nn.Linear(self.n_heads * self.head_dim,
                                config.hidden_size,
                                bias=False)

        self.dropout = config.dropout
        # Bảng RoPE dùng chung cho mọi lần forward.
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len,
                                    config.rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Nhân bản key/value head để khớp số lượng query head.
        x: (batch, n_kv_heads, seq_len, head_dim)
        -> (batch, n_heads, seq_len, head_dim)
        """
        b, n_kv, s, d = x.shape
        if self.n_rep == 1:
            return x
        # Chèn một chiều rồi expand (không copy bộ nhớ) rồi reshape.
        x = x[:, :, None, :, :].expand(b, n_kv, self.n_rep, s, d)
        return x.reshape(b, n_kv * self.n_rep, s, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        Trả về: (batch, seq_len, hidden_size)
        """
        b, s, _ = x.shape

        # 1) Chiếu ra Q, K, V rồi tách thành nhiều head.
        # (b, s, n_heads*head_dim) -> (b, n_heads, s, head_dim)
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2) Áp dụng RoPE lên Q và K (V không xoay).
        cos, sin = self.rope(s)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        # 3) Nhân bản K, V cho đủ số query head (đặc trưng GQA).
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 4) Scaled dot-product attention với mask nhân quả.
        # is_causal=True: token chỉ được nhìn các token trước nó.
        # PyTorch tự chọn kernel nhanh nhất (FlashAttention nếu có).
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )

        # 5) Gộp các head lại và chiếu về hidden_size.
        # (b, n_heads, s, head_dim) -> (b, s, n_heads*head_dim)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)
