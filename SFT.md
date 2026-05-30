# ĐẶC TẢ TINH CHỈNH GIÁM SÁT (SFT SPECIFICATION BLUEPRINT) - BIGRAM V2 1.8B
*Dành cho AI Agent thực thi quá trình SFT căn chỉnh hành vi mô hình.*

Tài liệu này cung cấp các hướng dẫn kỹ thuật chi tiết để chạy tinh chỉnh giám sát (Supervised Fine-Tuning - SFT) cho mô hình **Bigram V2 1.8B** sau khi đã hoàn tất pretrain.

---

## 📂 1. ĐẶC TẢ DỮ LIỆU ĐẦU VÀO SFT

### Định dạng File:
* Dữ liệu SFT là tệp `sft.jsonl` và `sft_val.jsonl` nằm trong thư mục `data/`.
* Mỗi dòng là một đối tượng JSON chứa `prompt` và `response` (được chưng cất từ Claude theo đặc tả của `SFT-data.md`).

### Cơ chế Loss Masking (Mặt nạ Loss SFT):
Để tránh mô hình bị suy giảm khả năng tiền huấn luyện và học thuộc lòng câu hỏi, SFT Dataset của Bigram áp dụng cơ chế mặt nạ:
* **Prompt**: Gán nhãn đích (targets) bằng `-100` (được bỏ qua bởi PyTorch cross-entropy).
* **Response**: Tính loss bình thường trên các token phản hồi của mô hình.

---

## ⚡ 2. SIÊU THAM SỐ TINH CHỈNH (SFT HYPERPARAMETERS)

Khi chạy SFT, AI Agent cấu hình các tham số tinh chỉnh nhẹ để tránh làm biến dạng tri thức nền tảng (catastrophic forgetting):

* `learning_rate = 5e-5` (LR nhỏ hơn nhiều so với mức pretrain `3e-4`)
* `weight_decay = 0.05`
* `dropout = 0.1` (Tăng nhẹ để tránh overfitting trên dữ liệu SFT nhỏ)
* `epochs = 3` (Chỉ chạy từ 2 đến 3 epochs để giữ tính tự nhiên)
* `batch_size = 4`
* `grad_accum_steps = 16`
* `warmup_steps = 500`
* `use_amp = True` (Mixed precision BF16)
* `gradient_checkpointing = True`

---

## 🚀 3. LỆNH CHẠY SFT (SFT EXECUTION COMMAND)

AI Agent thực thi câu lệnh sau để bắt đầu SFT, chỉ định đường dẫn nạp trọng số đã pretrain qua tham số `--init`:

```bash
python scripts/train_sft.py \
  --data data/sft.jsonl \
  --val-data data/sft_val.jsonl \
  --tokenizer data/tokenizer_v2.json \
  --init checkpoints/pretrain/ckpt_final.pt \
  --out-dir checkpoints/sft \
  --max-steps 10000
```

---

## 🧪 4. KIỂM CHỨNG & ĐÁNH GIÁ (SFT VALIDATION)

### 1. Suy luận không gian ẩn (System 2 test)
Sau khi SFT hoàn tất, chạy thử nghiệm kiểm tra tính năng sinh suy luận ẩn:
```bash
python scripts/generate.py \
  --checkpoint checkpoints/sft/ckpt_final.pt \
  --tokenizer data/tokenizer_v2.json \
  --prompt "Nếu hôm qua là thứ hai thì ngày mai là thứ mấy? Suy nghĩ kỹ." \
  --recurrence 16 \
  --temperature 0.5
```

### 2. Tiêu chuẩn đánh giá chất lượng:
* Mô hình sinh ra thẻ `<think>` và thực hiện lập luận từng bước logic bằng tiếng Việt trước khi đưa ra câu trả lời chính xác.
* Mô hình gọi tool chuẩn xác bằng cú pháp `<tool_call> ... </tool_call>` khi gặp các câu hỏi tính toán phức tạp hoặc chạy code hệ thống.
* SFT Validation loss cần tiệm cận $\sim 1.0 - 1.5$ và không có hiện tượng nổ gradient.
