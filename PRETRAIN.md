# ĐẶC TẢ QUY TRÌNH TIỀN HUẤN LUYỆN (PRETRAIN BLUEPRINT) - BIGRAM V2 1.8B
*Dành cho AI Agent thực thi quá trình huấn luyện từ đầu trên 1 GPU RTX A6000 (48GB VRAM).*

Tài liệu này cung cấp các hướng dẫn kỹ thuật chi tiết, câu lệnh thực thi và siêu tham số để chạy pretrain siêu lõi **Bigram V2 1.8B** sử dụng cấu hình preset vàng `bigram_v2_1_8b_a6000_config`.

---

## 📅 1. CHUẨN BỊ KHO DỮ LIỆU HUÂN LUYỆN (DATA PREPARATION)

### Yêu cầu về Volume & Chất lượng:
* **Quy mô mục tiêu**: 100 Tỷ đến 200 Tỷ tokens.
* **Tỷ lệ trộn dữ liệu (Data Mix)**:
  * **20% Dữ liệu chưng cất (Claude Distilled)**: Dạng textbook giả lập khoa học tự nhiên, toán logic CoT và code giải thuật sạch.
  * **50% Dữ liệu tiếng Việt thô chất lượng cao**: Đã qua NFC normalization, lọc bỏ HTML tags, lọc ngôn ngữ rác.
  * **30% Dữ liệu mã nguồn & Tài liệu kỹ thuật tiếng Anh**: Đọc hiểu từ GitHub, arXiv chuyên ngành.

### Bước 1: Huấn luyện Tokenizer VS-BPE mới
Chạy script huấn luyện tokenizer đơn luồng nhận biết âm tiết tiếng Việt trên toàn bộ kho ngữ liệu thô:
```bash
python scripts/train_tokenizer.py \
  --input data/corpus.txt \
  --output data/tokenizer_v2.json \
  --vocab-size 32000 \
  --min-frequency 2 \
  --tokenizer-type vs_bpe
```

### Bước 2: Nén nhị phân dữ liệu (Data Pack)
Chuyển đổi các file văn bản `.txt` thành định dạng nhị phân `.bin` (uint16) để tăng tốc độ nạp dữ liệu vào GPU:
```bash
python scripts/prepare_data.py \
  --tokenizer data/tokenizer_v2.json \
  --input data/train.txt \
  --output-prefix data/train

python scripts/prepare_data.py \
  --tokenizer data/tokenizer_v2.json \
  --input data/val.txt \
  --output-prefix data/val
```

---

## ⚡ 2. ĐẶC TẢ SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)

AI Agent khi chạy script huấn luyện phải nạp cấu hình `bigram_v2_1_8b_a6000_config` hoặc cấu hình các thông số sau:

### Thông số Mô hình (Model Config)
* `vocab_size = 32000`
* `hidden_size = 2048`
* `intermediate_size = 5504`
* `max_seq_len = 16384` (16k context, tương đương ~48k Llama tokens)
* `n_prelude_layers = 2`
* `n_recurrent_layers = 6` (Đan xen 3 layer MLA + 3 layer Mamba-2 SSM thuần PyTorch)
* `n_coda_layers = 2`
* `use_mla = True` (`kv_latent_dim = 128`, `decoupled_rope_dim = 64`)
* `use_pondernet = True` (`pondernet_prior_p = 0.3`)
* `use_mamba = True`
* `use_moe = True` (`n_experts = 8`, `n_experts_active = 2`, `use_vietnamese_expert = True`)
* `tie_embeddings = True`

### Thông số Huấn luyện (Train Config)
* `batch_size = 2` (trên mỗi GPU)
* `grad_accum_steps = 64` (Effective batch size = 128 sequences)
* `learning_rate = 3e-4` (Cosine learning rate decay với `min_lr_ratio = 0.1`)
* `weight_decay = 0.1`
* `adam_beta1 = 0.9`, `adam_beta2 = 0.95`
* `grad_clip = 1.0`
* `warmup_steps = 1500`
* `use_amp = True` (Sử dụng mixed precision BF16 để tối ưu Tensor Cores trên A6000)
* `gradient_checkpointing = True` (Tiết kiệm VRAM kích hoạt)
* `compile_model = True` (Sử dụng `torch.compile` để tối ưu hóa kernel của PyTorch 2.x)

---

## 🚀 3. LỆNH CHẠY TIỀN HUẤN LUYỆN (PRETRAIN RUN COMMAND)

AI Agent kích hoạt tiến trình pretrain bằng câu lệnh sau:
```bash
python scripts/train.py \
  --train-data data/train \
  --val-data data/val \
  --tokenizer data/tokenizer_v2.json \
  --config configs/bigram_v2_1_8b.json \
  --max-steps 150000 \
  --out-dir checkpoints/pretrain
```
*(Lưu ý: Nếu chưa ghi config ra JSON, viết một script nhỏ hoặc cập nhật config.py để chọn preset `bigram_v2_1_8b_a6000_config` trực tiếp).*

---

## ⚠️ 4. GIÁM SÁT & XỬ LÝ SỰ CỐ (LOGGING & TROUBLESHOOTING)

### 1. Giám sát Hàm Mất Mát (Loss Tracking)
* **LM Loss**: Cần giảm đều từ $\sim 10.0$ ban đầu xuống dưới $\sim 2.5$ ở cuối quá trình pretrain.
* **Halting Loss (PonderNet)**: Cần hội tụ và ổn định quanh mức $\sim 0.5 - 1.2$. Nếu Halting Loss bùng nổ lên $> 5.0$, hãy kiểm tra hệ số `pondernet_prior_p` và giảm nhẹ learning rate.
* **MoE Aux Loss**: Đảm bảo tải trọng của experts được phân phối cân bằng tự nhiên nhờ thuật toán Auxiliary-Loss-Free Gating.

### 2. Xử lý Tràn bộ nhớ (Out Of Memory - OOM)
Nếu card A6000 báo lỗi OOM:
1. Đảm bảo đã bật `gradient_checkpointing = True`.
2. Chuyển đổi optimizer sang dạng **8-bit AdamW** (bằng cách sửa `trainer.py` import `bitsandbytes.optim.Adam8bit`).
3. Giảm nhẹ `batch_size = 1` và tăng `grad_accum_steps = 128` để giữ nguyên effective batch size.
