"""
ffn.py
======
Tầng feed-forward (FFN) của Bigram. Có hai lựa chọn:

  1. SwiGLUMlp : MLP dày thông thường dùng activation SwiGLU.
  2. MoEFFN    : Mixture of Experts — chỉ kích hoạt vài expert mỗi token.

MoE cho phép tăng "dung lượng" model (nhiều tham số) mà chi phí tính toán
mỗi token gần như không đổi (chỉ chạy top-k expert thay vì tất cả).

Điểm đặc trưng Bigram: MoEFFN có thêm tùy chọn "Vietnamese expert" — một
expert LUÔN được kích hoạt, dành riêng để tích lũy tri thức tiếng Việt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMlp(nn.Module):
    """
    MLP với activation SwiGLU (dùng trong Llama, PaLM, Huginn).

        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Nhánh "gate" đóng vai trò cổng điều tiết nhánh "up" — thực nghiệm cho
    thấy hiệu quả hơn ReLU/GELU thuần.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """
    Mixture of Experts feed-forward.

    Cách hoạt động cho từng token:
      1. Một "router" (lớp Linear nhỏ) chấm điểm tất cả expert.
      2. Chọn top-k expert điểm cao nhất.
      3. Cho token đi qua k expert đó, trộn kết quả theo trọng số softmax.
      4. (Tùy chọn) cộng thêm kết quả của "Vietnamese expert" luôn-bật.

    Ngoài ra trả về một "auxiliary loss" để cân bằng tải giữa các expert
    (tránh tình trạng router chỉ dồn vào một vài expert — gọi là routing collapse).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_experts = config.n_experts
        self.n_active = config.n_experts_active
        self.use_vn_expert = config.use_vietnamese_expert
        self.aux_coef = config.moe_aux_loss_coef

        # Router: chiếu hidden -> điểm số cho từng expert.
        self.router = nn.Linear(config.hidden_size, config.n_experts, bias=False)

        # Danh sách các expert "cạnh tranh nhau".
        self.experts = nn.ModuleList([
            SwiGLUMlp(config.hidden_size, config.intermediate_size)
            for _ in range(config.n_experts)
        ])

        # Expert tiếng Việt luôn-bật (nếu được kích hoạt trong config).
        if self.use_vn_expert:
            self.vn_expert = SwiGLUMlp(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, hidden_size)
        Trả về: (output cùng shape với x, aux_loss là scalar)
        """
        b, s, h = x.shape
        # Gộp batch và seq lại để xử lý token-độc-lập: (N, h) với N = b*s.
        x_flat = x.view(-1, h)
        n_tokens = x_flat.shape[0]

        # --- Bước 1: router chấm điểm ---
        router_logits = self.router(x_flat)               # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)    # (N, n_experts)

        # --- Bước 2: chọn top-k expert ---
        # topk_probs: trọng số; topk_idx: chỉ số expert được chọn.
        topk_probs, topk_idx = torch.topk(router_probs, self.n_active, dim=-1)
        # Chuẩn hóa lại để tổng trọng số của k expert = 1.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # --- Bước 3: cho token qua các expert được chọn ---
        out_flat = torch.zeros_like(x_flat)
        # Lặp qua từng expert; với mỗi expert, xử lý mọi token đã chọn nó.
        for e_idx in range(self.n_experts):
            # mask: token nào (trong N token) có chọn expert e_idx?
            # (topk_idx == e_idx) -> (N, k) bool; any theo k -> (N,) bool.
            sel = (topk_idx == e_idx)
            token_mask = sel.any(dim=-1)
            if not token_mask.any():
                continue  # Không token nào chọn expert này -> bỏ qua.
            # Lấy các token tương ứng, cho qua expert.
            tokens = x_flat[token_mask]
            expert_out = self.experts[e_idx](tokens)
            # Lấy trọng số gating của expert này cho các token đó.
            # sel[token_mask] -> (M, k); nhân với topk_probs rồi tổng theo k.
            weight = (topk_probs[token_mask] * sel[token_mask]).sum(dim=-1, keepdim=True)
            # Cộng đóng góp có trọng số vào output.
            out_flat[token_mask] += expert_out * weight

        # --- Bước 4: Vietnamese expert luôn-bật ---
        if self.use_vn_expert:
            out_flat = out_flat + self.vn_expert(x_flat)

        # --- Auxiliary loss: cân bằng tải giữa các expert ---
        # Ý tưởng (theo Switch Transformer): phạt khi phân phối token giữa
        # các expert lệch khỏi đều. f_i = tỉ lệ token đi vào expert i;
        # P_i = xác suất router trung bình cho expert i.
        # aux = n_experts * sum(f_i * P_i).
        if self.training:
            # Tỉ lệ token thực sự được route tới mỗi expert.
            expert_mask = F.one_hot(topk_idx, self.n_experts).float()  # (N, k, n_experts)
            f_i = expert_mask.sum(dim=(0, 1)) / (n_tokens * self.n_active)
            # Xác suất router trung bình.
            p_i = router_probs.mean(dim=0)
            aux_loss = self.aux_coef * self.n_experts * torch.sum(f_i * p_i)
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return out_flat.view(b, s, h), aux_loss
