"""
config.py
=========
Định nghĩa toàn bộ siêu tham số (hyperparameter) cho Bigram.

Triết lý thiết kế: TẤT CẢ cấu hình tập trung vào một chỗ duy nhất, dưới dạng
các dataclass có giá trị mặc định hợp lý. Mọi script (train, eval, ...) đều
nhận một đối tượng `BigramConfig` thay vì truyền lẻ tẻ từng tham số. Nhờ vậy
thí nghiệm có thể tái lập (reproducible) và dễ lưu lại.

Kiến trúc Bigram dựa trên 3 trụ cột (xem PHILOSOPHY.md):
  1. Recurrent-depth: khối lõi được lặp r vòng -> suy luận sâu mà ít tham số.
  2. Abstention head: một đầu ra riêng để model "biết khi nào nên im lặng".
  3. Tonal-aware tokenizer: tách âm gốc và thanh điệu tiếng Việt thành 2 luồng.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ModelConfig:
    """Siêu tham số cho kiến trúc model (phần mạng nơ-ron)."""

    # --- Kích thước cơ bản ---
    vocab_size: int = 32000          # Số token trong từ điển (luồng âm gốc).
    tone_vocab_size: int = 8         # Số thanh điệu tiếng Việt: 6 thanh + <none> + <pad>.
    hidden_size: int = 512           # Chiều của vector ẩn (h). Model nhỏ: 512.
    intermediate_size: int = 1376    # Chiều ẩn của MLP (thường ~2.67x hidden_size cho SwiGLU).
    max_seq_len: int = 1024          # Độ dài chuỗi tối đa.

    # --- Cấu trúc Prelude / Recurrent / Coda ---
    # Bộ ba (lP, lR, lC): số layer trong từng nhóm. Theo paper Huginn (2,4,2).
    n_prelude_layers: int = 2        # lP: số layer đưa input vào không gian latent.
    n_recurrent_layers: int = 4      # lR: số layer trong khối lõi được LẶP LẠI.
    n_coda_layers: int = 2           # lC: số layer giải mã latent -> dự đoán.

    # --- Attention ---
    n_heads: int = 8                 # Số attention head cho query.
    n_kv_heads: int = 2              # Số head cho key/value (GQA: n_kv_heads < n_heads).
    head_dim: Optional[int] = None   # Chiều mỗi head. None -> tự tính = hidden_size // n_heads.
    rope_theta: float = 50000.0      # Tần số cơ sở của RoPE (paper Huginn dùng 50000).
    attn_qk_bias: bool = True        # Huginn dùng bias học được cho Q và K (chỗ khác thì không).

    # --- Recurrence (đặc thù của kiến trúc recurrent-depth) ---
    mean_recurrence: float = 32.0    # r trung bình khi sample lúc train (phân phối log-normal Poisson).
    recurrence_sigma: float = 0.5    # Độ lệch chuẩn của phân phối sample r.
    backprop_depth: int = 8          # k: chỉ backprop qua k vòng cuối (truncated BPTT).
    state_init_std: float = 1.0      # Độ lệch chuẩn khi khởi tạo latent state s0 ~ N(0, std^2).
    state_init_mode: str = "zeros"   # "zeros" = deterministic latent start, "normal" = legacy random start.
    recurrent_early_exit_tol: float = 0.0  # Eval-only: stop recurrence when state delta falls below this.
    recurrent_early_exit_min_steps: int = 2  # Minimum recurrent steps before early-exit can trigger.

    # --- Mixture of Experts (MoE) trong khối recurrent ---
    use_moe: bool = True             # Bật/tắt MoE. Tắt -> dùng MLP dày thông thường.
    moe_scope: str = "all"           # "all" | "recurrent_only" | "none".
    n_experts: int = 8               # Số expert "cạnh tranh" trong router.
    n_experts_active: int = 2        # Số expert được chọn mỗi token (top-k routing).
    use_vietnamese_expert: bool = True  # Thêm 1 expert luôn-bật dành riêng cho token tiếng Việt.
    moe_aux_loss_coef: float = 0.01  # Hệ số cho load-balancing loss của MoE.

    # --- Abstention head (trụ cột chống hallucination) ---
    use_abstention_head: bool = True # Bật đầu ra "tôi không chắc".
    abstention_loss_coef: float = 0.1  # Hệ số cho loss của abstention head.

    # --- Tone head (đầu ra luồng thanh điệu) ---
    # Tokenizer của Bigram nhận vào HAI luồng (âm gốc + thanh điệu), nên đầu
    # ra cũng có hai luồng tương ứng: lm_head dự đoán âm gốc, tone_head dự
    # đoán thanh điệu. Nhờ vậy văn bản sinh ra GIỮ ĐƯỢC DẤU THANH.
    use_tone_head: bool = True       # Bật đầu ra dự đoán thanh điệu.
    tone_loss_coef: float = 0.5      # Hệ số cho loss của tone head.

    # --- Tool/verifier heads cho Bigram Tensor 1 ---
    use_tool_head: bool = False      # Dự đoán ý định gọi tool, không execute tool trong model.
    n_tools: int = 32                # Kích thước từ điển tool name.
    tool_loss_coef: float = 0.3      # Hệ số loss tool routing/name.
    citation_loss_coef: float = 0.2  # Dành cho calibration citation/verifier.
    use_verifier_head: bool = False  # Dự đoán claim/source support score.
    verifier_loss_coef: float = 0.2  # Hệ số loss verifier.
    max_tool_json_len: int = 2048    # Giới hạn render/parse JSON tool-call ở runtime.

    # --- Khởi tạo & regularization ---
    init_std: float = 0.02           # Độ lệch chuẩn khi khởi tạo trọng số (kiểu GPT-2).
    norm_eps: float = 1e-5           # Epsilon cho RMSNorm (tránh chia cho 0).
    embedding_scale: bool = True     # Nhân embedding với sqrt(hidden_size) (paper Huginn dùng).
    dropout: float = 0.0             # Dropout. Mặc định 0 cho pretraining quy mô lớn.
    layerscale_init: float = 0.1     # Giá trị khởi tạo của LayerScale (chống latent collapse).
    tie_embeddings: bool = True      # Dùng chung ma trận embedding cho cả input và output head.

    def __post_init__(self):
        # Tự tính head_dim nếu người dùng không chỉ định.
        if self.head_dim is None:
            assert self.hidden_size % self.n_heads == 0, \
                "hidden_size phải chia hết cho n_heads"
            self.head_dim = self.hidden_size // self.n_heads
        # Kiểm tra ràng buộc GQA: số query head phải chia hết cho số kv head.
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads phải chia hết cho n_kv_heads (ràng buộc của GQA)"
        # Kiểm tra MoE.
        assert self.moe_scope in {"all", "recurrent_only", "none"}, \
            "moe_scope phải là 'all', 'recurrent_only', hoặc 'none'"
        if self.moe_scope == "none":
            self.use_moe = False
        if self.use_moe:
            assert self.n_experts_active <= self.n_experts, \
                "Số expert active không thể lớn hơn tổng số expert"
        if self.use_tool_head:
            assert self.n_tools > 0, "n_tools phải > 0 khi bật tool head"
        assert self.state_init_mode in {"zeros", "normal"}, \
            "state_init_mode phải là 'zeros' hoặc 'normal'"
        assert self.recurrent_early_exit_tol >= 0.0, \
            "recurrent_early_exit_tol không được âm"
        assert self.recurrent_early_exit_min_steps >= 1, \
            "recurrent_early_exit_min_steps phải >= 1"


@dataclass
class TrainConfig:
    """Siêu tham số cho quá trình huấn luyện."""

    # --- Optimizer (AdamW, theo paper MoDr/Huginn) ---
    learning_rate: float = 4e-4      # LR đỉnh. Model nhỏ có thể dùng cao hơn 4e-5 của paper.
    weight_decay: float = 0.1        # Trọng số phạt L2.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95         # Paper Huginn dùng 0.95 (thấp hơn mặc định 0.999).
    adam_eps: float = 1e-8
    grad_clip: float = 1.0           # Ngưỡng clip gradient. Recurrent cần clip chặt (paper: 0.2).

    # --- Lịch học (learning rate schedule) ---
    warmup_steps: int = 100          # Số bước tăng dần LR từ 0 lên đỉnh.
    max_steps: int = 10000           # Tổng số bước train.
    min_lr_ratio: float = 0.1        # LR cuối = min_lr_ratio * learning_rate (cosine decay).

    # --- Batch ---
    batch_size: int = 8              # Số chuỗi mỗi batch (trên mỗi thiết bị).
    grad_accum_steps: int = 4        # Tích lũy gradient -> batch hiệu dụng = batch_size * accum.

    # --- Khoảng thời gian (interval) ---
    log_interval: int = 10           # Cứ mỗi N bước thì in log.
    eval_interval: int = 500         # Cứ mỗi N bước thì chạy đánh giá.
    save_interval: int = 1000        # Cứ mỗi N bước thì lưu checkpoint.

    # --- Kỹ thuật tiết kiệm bộ nhớ ---
    use_amp: bool = True             # Dùng mixed precision (bf16/fp16) nếu có GPU.
    gradient_checkpointing: bool = False  # Đánh đổi tốc độ lấy bộ nhớ.

    # --- Khác ---
    seed: int = 1337                 # Hạt giống ngẫu nhiên (để tái lập thí nghiệm).
    device: str = "auto"             # "auto" -> tự chọn cuda nếu có, ngược lại cpu.
    compile_model: bool = False      # torch.compile (nhanh hơn nhưng cần PyTorch 2.x ổn định).
    out_dir: str = "checkpoints"     # Thư mục lưu checkpoint.


@dataclass
class DataConfig:
    """Cấu hình cho dữ liệu huấn luyện."""

    data_dir: str = "data"           # Thư mục chứa dữ liệu.
    train_file: str = "train.bin"    # File dữ liệu train (token đã mã hóa, dạng nhị phân uint16).
    val_file: str = "val.bin"        # File dữ liệu validation.
    tokenizer_file: str = "tokenizer.json"  # File tokenizer đã train.

    # Cấu hình giai đoạn (stage). Pipeline có 5 giai đoạn: xem PHILOSOPHY.md.
    # "pretrain" | "midtrain" | "sft" | "dpo" | "calibration"
    stage: str = "pretrain"


@dataclass
class BigramConfig:
    """Config tổng — gom 3 nhóm config con lại."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: str):
        """Lưu config ra file JSON để tái lập thí nghiệm sau này."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BigramConfig":
        """Đọc config từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**d["train"]),
            data=DataConfig(**d["data"]),
        )


# Một vài cấu hình dựng sẵn để tiện dùng ----------------------------------

def tiny_config() -> BigramConfig:
    """Cấu hình cực nhỏ — dùng để chạy test nhanh và debug trên CPU."""
    cfg = BigramConfig()
    cfg.model.vocab_size = 256
    cfg.model.hidden_size = 64
    cfg.model.intermediate_size = 172
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.n_prelude_layers = 1
    cfg.model.n_recurrent_layers = 2
    cfg.model.n_coda_layers = 1
    cfg.model.max_seq_len = 128
    cfg.model.mean_recurrence = 4.0
    cfg.model.backprop_depth = 3
    cfg.model.n_experts = 4
    cfg.train.batch_size = 4
    cfg.train.max_steps = 50
    return cfg


def small_config() -> BigramConfig:
    """Cấu hình ~0.5B tham số — 'Bigram-small', proof-of-concept thực sự."""
    cfg = BigramConfig()
    cfg.model.vocab_size = 32000
    cfg.model.hidden_size = 1024
    cfg.model.intermediate_size = 2752
    cfg.model.n_heads = 16
    cfg.model.n_kv_heads = 4
    cfg.model.n_prelude_layers = 2
    cfg.model.n_recurrent_layers = 4
    cfg.model.n_coda_layers = 2
    cfg.model.max_seq_len = 2048
    return cfg


def tensor1_config() -> BigramConfig:
    """Cấu hình Bigram Tensor 1 khoảng 1B params cho 1 GPU 48GB."""
    cfg = BigramConfig()
    cfg.model.vocab_size = 64000
    cfg.model.tone_vocab_size = 8
    cfg.model.hidden_size = 2048
    cfg.model.intermediate_size = 5504
    cfg.model.max_seq_len = 4096
    cfg.model.n_prelude_layers = 2
    cfg.model.n_recurrent_layers = 4
    cfg.model.n_coda_layers = 2
    cfg.model.n_heads = 32
    cfg.model.n_kv_heads = 8
    cfg.model.head_dim = 64
    cfg.model.rope_theta = 1000000.0
    cfg.model.mean_recurrence = 24.0
    cfg.model.recurrence_sigma = 0.7
    cfg.model.backprop_depth = 4
    cfg.model.state_init_mode = "zeros"
    cfg.model.recurrent_early_exit_tol = 0.0
    cfg.model.recurrent_early_exit_min_steps = 4
    cfg.model.use_moe = True
    cfg.model.moe_scope = "recurrent_only"
    cfg.model.n_experts = 4
    cfg.model.n_experts_active = 2
    cfg.model.use_vietnamese_expert = True
    cfg.model.use_abstention_head = True
    cfg.model.use_tone_head = True
    cfg.model.use_tool_head = True
    cfg.model.use_verifier_head = True
    cfg.model.dropout = 0.0
    cfg.model.tie_embeddings = True

    cfg.train.batch_size = 1
    cfg.train.grad_accum_steps = 128
    cfg.train.learning_rate = 2e-4
    cfg.train.weight_decay = 0.1
    cfg.train.warmup_steps = 2000
    cfg.train.grad_clip = 1.0
    cfg.train.use_amp = True
    cfg.train.gradient_checkpointing = True
    cfg.train.compile_model = False

    cfg.data.stage = "pretrain"
    cfg.model.__post_init__()
    return cfg
