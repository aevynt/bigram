"""
layers.py
=========
Các thành phần nền tảng (building block) dùng lại nhiều lần trong Bigram:
  - RMSNorm        : chuẩn hóa, nhẹ và ổn định hơn LayerNorm.
  - RotaryEmbedding: mã hóa vị trí kiểu RoPE (xoay vector query/key).
  - LayerScale     : nhân output mỗi layer với một scalar học được rất nhỏ;
                     đây là chốt chống "latent collapse" khi lặp khối recurrent.

Các hàm helper cho RoPE cũng nằm ở đây.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Khác LayerNorm ở chỗ KHÔNG trừ trung bình, chỉ chia cho căn bậc hai của
    trung bình bình phương. Ít phép tính hơn, và thực nghiệm cho thấy ổn
    định tương đương hoặc tốt hơn cho transformer (dùng trong Llama, Huginn).

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # weight khởi tạo bằng 1 -> ban đầu lớp norm gần như không đổi tín hiệu.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tính ở float32 để tránh sai số khi train mixed-precision, rồi ép lại kiểu cũ.
        dtype = x.dtype
        x = x.float()
        # rsqrt = 1/sqrt. keepdim=True để broadcast lại đúng chiều.
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


class LayerScale(nn.Module):
    """
    LayerScale: nhân đầu ra của một nhánh (attention hoặc MLP) với một vector
    scalar học được, khởi tạo bằng giá trị rất nhỏ (ví dụ 0.1).

    Vì sao cần cho Bigram: khối recurrent được lặp hàng chục lần. Nếu mỗi vòng
    cộng vào residual stream một lượng lớn, latent state sẽ "nổ" hoặc "sụp"
    (latent collapse). Khởi tạo LayerScale nhỏ khiến mỗi vòng lặp ban đầu chỉ
    tinh chỉnh latent một chút -> recurrence ổn định, model tự học tăng dần
    mức đóng góp nếu cần.
    """

    def __init__(self, dim: int, init_value: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Thay vì cộng một vector vị trí vào embedding, RoPE "xoay" từng cặp chiều
    của vector query/key một góc tỉ lệ với vị trí token. Tích vô hướng giữa
    query và key sau khi xoay chỉ phụ thuộc vào KHOẢNG CÁCH tương đối giữa
    hai token -> model tổng quát hóa tốt ra chuỗi dài.

    Lớp này chỉ tính sẵn bảng cos/sin. Việc xoay thực sự nằm ở hàm
    `apply_rotary` bên dưới.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 50000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim phải chẵn để RoPE chia cặp được"
        self.head_dim = head_dim

        # inv_freq[i] = 1 / theta^(2i/head_dim) — tần số cho từng cặp chiều.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Vị trí token: 0, 1, ..., max_seq_len-1.
        t = torch.arange(max_seq_len).float()
        # freqs[pos, i] = pos * inv_freq[i]  (ngoài tích = outer product).
        freqs = torch.outer(t, inv_freq)
        # Lặp đôi để khớp với việc tách cặp trong apply_rotary.
        emb = torch.cat([freqs, freqs], dim=-1)
        # register_buffer: lưu cùng model nhưng KHÔNG phải tham số học được.
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        """Trả về bảng cos/sin đã cắt đúng độ dài chuỗi hiện tại."""
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Hàm phụ cho RoPE: tách vector làm đôi theo chiều cuối, rồi ghép lại theo
    kiểu (-nửa_sau, nửa_đầu). Đây là phép xoay 90 độ trên từng cặp chiều.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    """
    Áp dụng phép xoay RoPE lên query và key.

    q, k có shape (batch, n_heads, seq_len, head_dim).
    cos, sin có shape (seq_len, head_dim) -> cần thêm 2 chiều để broadcast.

    Công thức:  x_rotated = x * cos + rotate_half(x) * sin
    """
    # (seq_len, head_dim) -> (1, 1, seq_len, head_dim) để broadcast theo batch & head.
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
