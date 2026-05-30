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

from .layers import RotaryEmbedding, apply_rotary, RMSNorm


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


class SSKVPAttention(GroupedQueryAttention):
    """GQA hỗ trợ chia sẻ Key-Value (SS-KVP) qua các vòng lặp dọc."""

    def __init__(self, config, shared_kv: bool = False):
        super().__init__(config)
        self.shared_kv = shared_kv
        self._cached_k = None
        self._cached_v = None

    def reset_kv_cache(self):
        """Xóa cache KV trước mỗi lần forward chính."""
        self._cached_k = None
        self._cached_v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        # 1) Chiếu Q.
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        # 2) Chiếu K, V (tính mới hoặc dùng cache).
        if self.shared_kv:
            if self._cached_k is None:
                k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
                self._cached_k = k
                self._cached_v = v
            else:
                k = self._cached_k
                v = self._cached_v
        else:
            k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3) Áp dụng RoPE lên Q và K (V không xoay).
        cos, sin = self.rope(s)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        # 4) Nhân bản K, V cho đủ số query head (đặc trưng GQA).
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 5) Scaled dot-product attention với mask nhân quả.
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )

        # 6) Gộp các head lại và chiếu về hidden_size.
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - DeepSeek-V3 Style.
    Nén KV cache thành một vector ẩn có chiều kích thước nhỏ giúp tiết kiệm 93% bộ nhớ KV.
    Tách biệt thông tin vị trí (Decoupled RoPE) để giữ tính chính xác.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.kv_latent_dim = getattr(config, "kv_latent_dim", 128)
        self.decoupled_rope_dim = getattr(config, "decoupled_rope_dim", 64)
        
        # --- KV Compression ---
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_latent_dim, bias=False)
        self.kv_down_norm = RMSNorm(self.kv_latent_dim, config.norm_eps)
        self.kv_up_proj = nn.Linear(self.kv_latent_dim, self.n_heads * (self.head_dim + self.decoupled_rope_dim), bias=False)
        
        # --- Q Compression ---
        self.q_down_proj = nn.Linear(self.hidden_size, self.kv_latent_dim, bias=False)
        self.q_down_norm = RMSNorm(self.kv_latent_dim, config.norm_eps)
        self.q_up_proj = nn.Linear(self.kv_latent_dim, self.n_heads * (self.head_dim + self.decoupled_rope_dim), bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = config.dropout
        # RoPE dùng cho phần decoupled position
        self.rope = RotaryEmbedding(self.decoupled_rope_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        
        # 1) Q Compression & Projection
        q_latent = self.q_down_norm(self.q_down_proj(x))
        q_up = self.q_up_proj(q_latent).view(b, s, self.n_heads, self.head_dim + self.decoupled_rope_dim).transpose(1, 2)
        # Tách phần content và phần position
        q_c, q_r = q_up.split([self.head_dim, self.decoupled_rope_dim], dim=-1)
        
        # 2) KV Compression & Projection
        kv_latent = self.kv_down_norm(self.kv_down_proj(x))
        kv_up = self.kv_up_proj(kv_latent).view(b, s, self.n_heads, self.head_dim + self.decoupled_rope_dim).transpose(1, 2)
        # Tách phần content và position của KV
        k_c, k_r = kv_up.split([self.head_dim, self.decoupled_rope_dim], dim=-1)
        
        # Tách V (chỉ là phần content, không xoay RoPE)
        v = k_c
        
        # 3) Áp dụng RoPE lên phần decoupled position (q_r và k_r)
        cos, sin = self.rope(s)
        cos = cos.to(q_r.dtype)
        sin = sin.to(q_r.dtype)
        q_r, k_r = apply_rotary(q_r, k_r, cos, sin)
        
        # 4) Ghép lại Q và K hoàn chỉnh (Content + Position)
        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r], dim=-1)
        
        # 5) Scaled dot-product attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )
        
        # 6) Gộp các head lại và chiếu về hidden_size
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)

