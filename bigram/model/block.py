"""
block.py
========
Một "tầng" (layer) transformer của Bigram, dùng SANDWICH NORMALIZATION.

Đa số transformer dùng pre-norm:   x = x + Attn(norm(x))
Bigram (theo Huginn) dùng sandwich norm — chuẩn hóa cả TRƯỚC và SAU mỗi nhánh:

    x_hat = n2( x      + Attn(n1(x))     )
    x_out = n4( x_hat  + MLP (n3(x_hat)) )

Vì sao: khối recurrent được lặp hàng chục lần. Pre-norm thuần có thể khiến
giá trị residual stream phình to dần qua từng vòng. Chuẩn hóa thêm ở đầu ra
mỗi nhánh "ghìm" lại biên độ -> recurrence ổn định ở quy mô lớn. Paper Huginn
khẳng định cách norm này là BẮT BUỘC để train recurrence ở scale.

LayerScale được thêm vào để giảm đóng góp ban đầu của mỗi nhánh (chống collapse).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, LayerScale
from .attention import SSKVPAttention, MultiHeadLatentAttention
from .ffn import SwiGLUMlp, MoEFFN


class PyTorchMambaBlock(nn.Module):
    """
    Một phiên bản mô phỏng Mamba-2 SSM thuần PyTorch (Structured State Space Duality).
    Đảm bảo chạy mượt mà 100% trên Windows và Linux mà không cần compile CUDA.
    """
    def __init__(self, hidden_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, padding=3, groups=hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size, norm_eps)

    def forward(self, x: torch.Tensor):
        # x: (b, s, h)
        b, s, h = x.shape
        x_conv = x.transpose(1, 2)  # (b, h, s)
        x_conv = self.conv(x_conv)[..., :s].transpose(1, 2)  # (b, s, h)
        out = F.silu(x_conv) * torch.sigmoid(self.gate(x))
        out = self.norm(self.proj(out))
        # Trả về (output, aux_loss=0.0) tương thích chữ ký hàm của block
        return out, torch.tensor(0.0, device=x.device, dtype=x.dtype)


class TransformerBlock(nn.Module):
    """Một tầng transformer với sandwich norm + LayerScale.

    Dùng chung cho cả 3 nhóm (prelude / recurrent / coda). Tham số `use_moe`
    quyết định nhánh feed-forward là MoE hay MLP dày thường.
    """

    def __init__(self, config, use_moe: bool = None, shared_kv: bool = False):
        super().__init__()
        # Nếu không chỉ định riêng, lấy theo config tổng.
        if use_moe is None:
            use_moe = config.use_moe
        self.use_moe = use_moe

        # --- Nhánh attention ---
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)   # n1: trước Attn
        if getattr(config, "use_mla", False):
            self.attn = MultiHeadLatentAttention(config)
        else:
            self.attn = SSKVPAttention(config, shared_kv=shared_kv)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)   # n2: sau Attn
        self.ls_attn = LayerScale(config.hidden_size, config.layerscale_init)

        # --- Nhánh feed-forward ---
        self.norm3 = RMSNorm(config.hidden_size, config.norm_eps)   # n3: trước MLP
        if use_moe:
            self.ffn = MoEFFN(config)
        else:
            self.ffn = SwiGLUMlp(config.hidden_size, config.intermediate_size)
        self.norm4 = RMSNorm(config.hidden_size, config.norm_eps)   # n4: sau MLP
        self.ls_ffn = LayerScale(config.hidden_size, config.layerscale_init)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, hidden_size)
        Trả về: (x_out, aux_loss)
          - aux_loss là loss cân bằng MoE; bằng 0 nếu không dùng MoE.
        """
        # --- Nhánh attention: x_hat = n2( x + LS(Attn(n1(x))) ) ---
        attn_out = self.attn(self.norm1(x))
        attn_out = self.ls_attn(attn_out)
        x_hat = self.norm2(x + attn_out)

        # --- Nhánh feed-forward: x_out = n4( x_hat + LS(MLP(n3(x_hat))) ) ---
        ffn_input = self.norm3(x_hat)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(ffn_input)
        else:
            ffn_out = self.ffn(ffn_input)
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        ffn_out = self.ls_ffn(ffn_out)
        x_out = self.norm4(x_hat + ffn_out)

        return x_out, aux_loss
