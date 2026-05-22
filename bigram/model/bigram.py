"""
bigram.py
=========
Mô hình Bigram hoàn chỉnh — kiến trúc transformer recurrent-depth.

Luồng dữ liệu (xem PHILOSOPHY.md, mục kiến trúc):

    token tiếng Việt  (âm gốc + thanh điệu)
            |
            v
    [Embedding]  -> cộng embedding âm gốc + embedding thanh điệu
            |
            v
    [PRELUDE]    -> lP layer, đưa input vào không gian latent (kết quả: e)
            |
            v
    s0 ~ N(0, std^2)                         (latent state khởi tạo ngẫu nhiên)
    si = RECURRENT(e, s_{i-1})   lặp r vòng   (khối lõi, được LẶP LẠI)
            |
            v
    [CODA]       -> lC layer, giải mã latent state cuối cùng
            |
       +----+----+
       v         v
   [LM Head]  [Abstention Head]
   token        P(model nên nói "tôi không chắc")

Hai cơ chế then chốt:
  - Truncated BPTT: forward chạy đủ r vòng, nhưng chỉ k vòng cuối có gradient.
    => bộ nhớ huấn luyện KHÔNG phụ thuộc r (cho phép r lớn mà vẫn train được).
  - Input injection: embedding `e` được đưa vào MỌI vòng lặp (không chỉ vòng
    đầu). Đây là điều kiện để recurrence ổn định và "path independent".
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .layers import RMSNorm
from .tooling import ToolHead


def sample_recurrence(mean_r: float, sigma: float,
                      generator: torch.Generator = None) -> int:
    """
    Sample số vòng lặp r từ phân phối log-normal Poisson (theo paper Huginn).

    Phân phối này hầu hết cho giá trị nhỏ hơn mean_r, nhưng thỉnh thoảng có
    "đuôi dài" cho giá trị lớn -> model thi thoảng được luyện suy luận rất sâu.

        tau ~ Normal( log(mean_r) - sigma^2/2 , sigma )
        r   ~ Poisson( exp(tau) ) + 1
    """
    mu = math.log(mean_r) - 0.5 * sigma ** 2
    tau = torch.normal(mean=mu, std=sigma, size=(1,), generator=generator)
    rate = torch.exp(tau)
    r = torch.poisson(rate, generator=generator).int().item() + 1
    return max(1, r)  # Đảm bảo ít nhất 1 vòng.


class BigramModel(nn.Module):
    """Mô hình ngôn ngữ Bigram với recurrent depth."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        h = config.hidden_size

        # --- Embedding ---
        # Luồng 1: âm gốc (ví dụ "nhung" trong "những").
        self.token_embedding = nn.Embedding(config.vocab_size, h)
        # Luồng 2: thanh điệu (sắc / huyền / hỏi / ngã / nặng / ngang).
        # Đây là điểm đặc trưng tiếng Việt — xem tokenizer.
        self.tone_embedding = nn.Embedding(config.tone_vocab_size, h)
        # Hệ số nhân embedding (gamma) — paper Huginn dùng sqrt(h).
        self.embed_scale = math.sqrt(h) if config.embedding_scale else 1.0

        # --- PRELUDE: lP layer đưa input vào latent space ---
        prelude_use_moe = config.use_moe and config.moe_scope == "all"
        self.prelude = nn.ModuleList([
            TransformerBlock(config, use_moe=prelude_use_moe)
            for _ in range(config.n_prelude_layers)
        ])

        # --- RECURRENT CORE: lR layer, được lặp lại nhiều vòng ---
        # Adapter: nhận [latent_state ; embedding] (ghép -> 2h) và chiếu về h.
        # Việc ghép (concat) thay vì cộng giúp ổn định hơn ở quy mô lớn (Huginn).
        recurrent_use_moe = (
            config.use_moe and config.moe_scope in {"all", "recurrent_only"}
        )
        self.recurrent_adapter = nn.Linear(2 * h, h, bias=False)
        self.recurrent = nn.ModuleList([
            TransformerBlock(config, use_moe=recurrent_use_moe)
            for _ in range(config.n_recurrent_layers)
        ])
        # RMSNorm ở cuối khối recurrent.
        self.recurrent_norm = RMSNorm(h, config.norm_eps)

        # --- CODA: lC layer giải mã latent state ---
        self.coda = nn.ModuleList([
            # Coda thường KHÔNG dùng MoE — giữ phần giải mã đơn giản, ổn định.
            TransformerBlock(config, use_moe=False)
            for _ in range(config.n_coda_layers)
        ])
        self.final_norm = RMSNorm(h, config.norm_eps)

        # --- Các đầu ra (head) ---
        # LM head: dự đoán token tiếp theo.
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        # Tie embeddings: dùng chung ma trận embedding cho input và output.
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Abstention head: dự đoán xác suất model NÊN từ chối / nói không chắc.
        # Đây là trụ cột chống hallucination — đầu ra nhị phân cho mỗi vị trí.
        if config.use_abstention_head:
            self.abstention_head = nn.Linear(h, 1, bias=True)

        # Tone head: dự đoán thanh điệu của token tiếp theo.
        # Cùng với lm_head (dự đoán âm gốc), đây là luồng thứ hai của đầu ra —
        # khớp với việc tokenizer nhận vào hai luồng. Nhờ tone head, văn bản
        # sinh ra giữ được dấu thanh tiếng Việt thay vì mất dấu.
        if config.use_tone_head:
            self.tone_head = nn.Linear(h, config.tone_vocab_size, bias=False)
        if config.use_tool_head:
            self.tool_head = ToolHead(config)
        if config.use_verifier_head:
            self.verifier_head = nn.Linear(h, 1, bias=True)

        # Khởi tạo trọng số.
        self.apply(self._init_weights)
        # Khởi tạo đặc biệt cho adapter: gần 0 ở phần embedding để vòng lặp đầu
        # tiên không bị nhiễu input lấn át -> recurrence khởi động êm.
        self._init_recurrent_stability()

    def _init_weights(self, module):
        """Khởi tạo trọng số kiểu GPT-2: Linear/Embedding ~ N(0, init_std)."""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_recurrent_stability(self):
        """
        Khởi tạo trọng số adapter nhỏ hơn để khối recurrent ổn định lúc đầu.
        Adapter có shape (h, 2h): nửa đầu nhân với latent state, nửa sau nhân
        với embedding. Ta scale nhỏ toàn bộ để bước nhảy mỗi vòng lặp ban đầu nhỏ.
        """
        with torch.no_grad():
            self.recurrent_adapter.weight.mul_(0.5)

    def num_parameters(self) -> int:
        """Đếm tổng số tham số học được (tiện cho việc báo cáo)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Các khối forward con
    # ------------------------------------------------------------------
    def embed(self, token_ids: torch.Tensor,
              tone_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Nhúng token thành vector. token_ids và tone_ids: (batch, seq_len).
        Nếu tone_ids = None thì coi như không có thông tin thanh điệu.
        """
        x = self.token_embedding(token_ids)
        if tone_ids is not None:
            x = x + self.tone_embedding(tone_ids)
        return x * self.embed_scale

    def run_prelude(self, x: torch.Tensor):
        """Chạy các layer prelude. Trả về (embedding latent e, aux_loss tổng)."""
        aux = 0.0
        for layer in self.prelude:
            if self.training and self.gradient_checkpointing:
                x, a = checkpoint(layer, x, use_reentrant=False)
            else:
                x, a = layer(x)
            aux = aux + a
        return x, aux

    def run_recurrent_step(self, e: torch.Tensor, state: torch.Tensor):
        """
        MỘT vòng lặp của khối recurrent: si = R(e, s_{i-1}).

        e     : embedding từ prelude (input injection - đưa vào mọi vòng).
        state : latent state của vòng trước.
        """
        # Ghép state và e theo chiều cuối -> (b, s, 2h), rồi adapter chiếu về h.
        combined = torch.cat([state, e], dim=-1)
        x = self.recurrent_adapter(combined)
        # Cho qua lR layer transformer.
        aux = 0.0
        for layer in self.recurrent:
            if self.training and self.gradient_checkpointing:
                x, a = checkpoint(layer, x, use_reentrant=False)
            else:
                x, a = layer(x)
            aux = aux + a
        return self.recurrent_norm(x), aux

    def run_coda(self, state: torch.Tensor):
        """Chạy các layer coda để giải mã latent state cuối cùng."""
        x = state
        aux = 0.0
        for layer in self.coda:
            if self.training and self.gradient_checkpointing:
                x, a = checkpoint(layer, x, use_reentrant=False)
            else:
                x, a = layer(x)
            aux = aux + a
        return self.final_norm(x), aux

    def init_recurrent_state(self, e: torch.Tensor,
                             generator: torch.Generator = None) -> torch.Tensor:
        """
        Initialize the recurrent latent state.

        The default ``zeros`` mode makes inference deterministic for a fixed
        prompt and sampling seed. ``normal`` keeps the original stochastic
        latent start for experiments that explicitly want it.
        """
        if self.config.state_init_mode == "normal":
            return torch.randn(
                e.shape,
                dtype=e.dtype,
                device=e.device,
                generator=generator,
            ) * self.config.state_init_std
        return torch.zeros_like(e)

    # ------------------------------------------------------------------
    # Forward chính
    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor,
                tone_ids: torch.Tensor = None,
                num_recurrence: int = None,
                generator: torch.Generator = None):
        """
        Forward pass đầy đủ.

        Tham số:
          token_ids      : (batch, seq_len) — id của âm gốc.
          tone_ids       : (batch, seq_len) — id thanh điệu (có thể None).
          num_recurrence : số vòng lặp r. Nếu None:
                             - lúc train -> sample ngẫu nhiên (random unrolling).
                             - lúc eval  -> dùng mean_recurrence.

        Trả về dict gồm:
          logits          : (batch, seq_len, vocab_size) — dự đoán token.
          abstention_logits: (batch, seq_len) — điểm "nên từ chối" (nếu bật head).
          aux_loss        : scalar — loss cân bằng MoE.
          num_recurrence  : số vòng lặp thực tế đã dùng.
        """
        cfg = self.config

        # --- Quyết định số vòng lặp r ---
        if num_recurrence is None:
            if self.training:
                # Random unrolling: mỗi forward lúc train dùng một r khác nhau.
                num_recurrence = sample_recurrence(
                    cfg.mean_recurrence, cfg.recurrence_sigma, generator)
            else:
                # Lúc eval: dùng giá trị trung bình cho ổn định.
                num_recurrence = int(round(cfg.mean_recurrence))
        r = max(1, num_recurrence)

        # --- Embedding ---
        x = self.embed(token_ids, tone_ids)

        # --- Prelude ---
        e, aux_total = self.run_prelude(x)

        # --- Khởi tạo latent state ---
        state = self.init_recurrent_state(e, generator)

        # --- Vòng lặp recurrent với TRUNCATED BACKPROP ---
        # Ý tưởng: chỉ k vòng CUỐI mới cần gradient. Các vòng trước chạy trong
        # torch.no_grad() -> không lưu activation -> tiết kiệm bộ nhớ, và bộ nhớ
        # này KHÔNG phụ thuộc vào r. Đây là chốt giúp train được r lớn.
        k = cfg.backprop_depth
        n_no_grad = max(0, r - k)   # số vòng đầu không cần gradient.

        # Phần 1: các vòng đầu — không gradient.
        if n_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_no_grad):
                    state, _ = self.run_recurrent_step(e, state)
            # Tách khỏi đồ thị tính toán (phòng hờ) — state giờ là "hằng số".
            state = state.detach()

        # Phần 2: k vòng cuối — CÓ gradient (sẽ được backprop).
        n_grad = r - n_no_grad
        steps_done = n_no_grad
        for _ in range(n_grad):
            prev_state = state
            state, aux = self.run_recurrent_step(e, state)
            aux_total = aux_total + aux
            steps_done += 1

            # Eval-only adaptive compute. If the recurrent latent state has
            # converged, stop spending extra recurrent steps. Training still
            # uses the full sampled depth so optimization remains predictable.
            if (not self.training
                    and cfg.recurrent_early_exit_tol > 0.0
                    and steps_done >= cfg.recurrent_early_exit_min_steps):
                delta = (state - prev_state).float().pow(2).mean().sqrt()
                if delta.item() <= cfg.recurrent_early_exit_tol:
                    break

        # --- Coda: giải mã ---
        decoded, aux = self.run_coda(state)
        aux_total = aux_total + aux

        # --- Các đầu ra ---
        logits = self.lm_head(decoded)

        out = {
            "logits": logits,
            "aux_loss": aux_total,
            "num_recurrence": r,
        }
        if cfg.use_abstention_head:
            # squeeze(-1): (b, s, 1) -> (b, s).
            out["abstention_logits"] = self.abstention_head(decoded).squeeze(-1)
        if cfg.use_tone_head:
            # (b, s, tone_vocab_size) — dự đoán thanh điệu token tiếp theo.
            out["tone_logits"] = self.tone_head(decoded)
        if cfg.use_tool_head:
            out.update(self.tool_head(decoded))
        if cfg.use_verifier_head:
            out["verifier_logits"] = self.verifier_head(decoded).squeeze(-1)

        return out

    # ------------------------------------------------------------------
    # Sinh văn bản (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor,
                 tone_ids: torch.Tensor = None,
                 max_new_tokens: int = 50,
                 num_recurrence: int = None,
                 temperature: float = 1.0,
                 top_k: int = None,
                 top_p: float = None,
                 repetition_penalty: float = 1.0,
                 abstention_threshold: float = None):
        """
        Sinh văn bản theo kiểu tự hồi quy (autoregressive).

        Tham số đáng chú ý:
          num_recurrence      : số vòng "suy nghĩ". Câu khó -> đặt cao hơn.
          abstention_threshold: nếu đặt và P(từ chối) vượt ngưỡng này, model
                                dừng lại — hiện thực hóa "biết khi nào nên im lặng".

        Trả về: (token_ids đã nối thêm, tone_ids tương ứng, có_từ_chối hay không).
        """
        self.eval()
        cfg = self.config
        abstained = False

        # Nếu chưa có tone_ids nhưng model có tone head, khởi tạo dãy tone 0
        # cùng độ dài prompt để có chỗ nối thanh điệu sinh ra.
        if tone_ids is None and cfg.use_tone_head:
            tone_ids = torch.zeros_like(token_ids)

        for _ in range(max_new_tokens):
            # Cắt bớt context nếu vượt quá độ dài tối đa.
            idx_cond = token_ids[:, -cfg.max_seq_len:]
            tone_cond = None
            if tone_ids is not None:
                tone_cond = tone_ids[:, -cfg.max_seq_len:]

            out = self.forward(idx_cond, tone_cond, num_recurrence=num_recurrence)
            logits = out["logits"][:, -1, :]  # chỉ lấy vị trí cuối.

            # Kiểm tra abstention: nếu model "thấy không chắc" thì dừng.
            if abstention_threshold is not None and "abstention_logits" in out:
                p_abstain = torch.sigmoid(out["abstention_logits"][:, -1])
                if (p_abstain > abstention_threshold).any():
                    abstained = True
                    break

            # Lấy mẫu token tiếp theo.
            logits = logits / max(temperature, 1e-6)
            if repetition_penalty and repetition_penalty != 1.0:
                for row, history in enumerate(token_ids):
                    seen = torch.unique(history)
                    logits[row, seen] = torch.where(
                        logits[row, seen] < 0,
                        logits[row, seen] * repetition_penalty,
                        logits[row, seen] / repetition_penalty,
                    )
            if top_k is not None:
                # Chỉ giữ top_k logit lớn nhất, phần còn lại gán -inf.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Nối token mới vào chuỗi.
            token_ids = torch.cat([token_ids, next_token], dim=1)
            if tone_ids is not None:
                if "tone_logits" in out:
                    # Model CÓ tone head -> dự đoán thanh điệu của token mới.
                    # Lấy argmax (thanh điệu ít cần "sáng tạo" — chọn chắc chắn).
                    tone_logits = out["tone_logits"][:, -1, :]
                    next_tone = tone_logits.argmax(dim=-1, keepdim=True)
                else:
                    # Không có tone head -> token mới gán tone 0 (<none>).
                    next_tone = torch.zeros_like(next_token)
                tone_ids = torch.cat([tone_ids, next_tone], dim=1)

        return token_ids, tone_ids, abstained
