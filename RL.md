# ĐẶC TẢ HỌC TĂNG CƯỜNG (RL SPECIFICATION BLUEPRINT) - BIGRAM V2 1.8B
*Dành cho AI Agent triển khai huấn luyện học tăng cường (Reinforcement Learning) căn chỉnh System 2.*

Tài liệu này cung cấp các đặc tả kỹ thuật và thiết kế hàm phần thưởng (Reward Functions) để chạy huấn luyện RL căn chỉnh khả năng unroll động và suy luận logic sâu sắc cho mô hình **Bigram V2 1.8B**.

---

## 🔬 1. THUẬT TOÁN ĐỀ XUẤT: GRPO (Group Relative Policy Optimization)

Thay vì sử dụng PPO truyền thống (yêu cầu một Critic Model cồng kềnh chiếm nhiều VRAM), Bigram V2 khuyến nghị sử dụng **GRPO** (thuật toán của DeepSeek-R1):
* **Cơ chế**: Sinh một nhóm $G$ phản hồi (ví dụ $G = 4$) cho cùng một prompt đầu vào.
* **Tối ưu**: Tính toán phần thưởng trung bình và độ lệch chuẩn của nhóm, tối ưu hóa Policy dựa trên điểm thưởng tương đối của các phản hồi trong nhóm. Giúp tiết kiệm **50% bộ nhớ GPU VRAM** so với PPO!

---

## 🏆 2. THIẾT KẾ CÁC HÀM PHẦN THƯỞNG (REWARD FUNCTIONS DESIGN)

AI Agent khi lập trình script huấn luyện RL phải triển khai 4 bộ lọc chấm điểm (Reward Filters) sau:

```
  Prompt ──► [Bigram V2 Policy] ──► 4 Phản hồi ──► [ Bộ Chấm Điểm RL ] ──► Tối ưu hóa Trọng số
                                                      │
                                                      ├─► 1. Format Reward (Thẻ <think>)
                                                      ├─► 2. Accuracy Reward (Giải đúng)
                                                      ├─► 3. Tool Calling Reward (JSON chuẩn)
                                                      └─► 4. PonderNet Efficiency Reward (Nghĩ tối ưu)
```

### 1. Format Reward (Phần thưởng Định dạng)
* **Quy tắc**: Phản hồi phải chứa đầy đủ và chính xác cấu trúc khối suy nghĩ.
* **Cách chấm**:
  * Có thẻ mở `<think>` ở đầu và thẻ đóng `</think>` trước câu trả lời: **+0.5 điểm**.
  * Có nội dung lập luận bên trong thẻ `<think>`: **+0.3 điểm**.
  * Định dạng JSON của tool call khớp schema: **+0.5 điểm**.

### 2. Accuracy Reward (Phần thưởng Độ chính xác)
* **Quy tắc**: Câu trả lời cuối cùng bên ngoài thẻ đóng `</think>` phải chính xác tuyệt đối.
* **Cách chấm**:
  * Trùng khớp đáp án toán học/logic mục tiêu: **+1.0 điểm**.
  * Trả lời sai tự tin: **-1.5 điểm** (Phạt cực nặng để triệt tiêu ảo giác).
  * Từ chối trả lời hợp lý thông qua Abstention Head ("Tôi không chắc"): **0.0 điểm** (Không phạt, khuyến khích tính trung thực).

### 3. Stateful Tool Calling Reward (Phần thưởng Gọi công cụ)
* **Quy tắc**: Khi gặp bài toán tính toán/lập trình phức tạp, mô hình phải ưu tiên gọi tool thay vì tự tính nhẩm.
* **Cách chấm**:
  * Có cấu trúc `<tool_call>` hợp lệ chứa JSON parse được: **+0.8 điểm**.
  * Gọi sai tên tool trong registry hoặc đối số không đúng schema: **-1.0 điểm**.

### 4. PonderNet Efficiency Reward (Phần thưởng Hiệu năng Ponder)
* **Quy tắc**: Khuyến khích mô hình nghĩ sâu cho toán khó, dừng sớm cho câu hỏi dễ thông qua việc tối ưu `halting_loss`.
* **Cách chấm**:
  * Tối ưu hóa hàm phạt KL Divergence giữa halting distribution thực tế và Prior Geometric distribution.
  * Phạt thêm nếu unroll quá sâu ($>24$ bước) cho các câu hỏi chào hỏi xã giao thông thường: **-0.5 điểm**.

---

## 🛠️ 3. KỊCH BẢN THỰC THI HUẤN LUYỆN (RL TRAINING SCRIPT)

AI Agent triển khai huấn luyện RL thông qua một script custom `scripts/train_rl.py` sử dụng thư viện **Ray / TRL (Transformer Reinforcement Learning)** của Hugging Face:

```bash
python scripts/train_rl.py \
  --policy-ckpt checkpoints/sft/ckpt_final.pt \
  --dataset data/reasoning_rl_dataset.jsonl \
  --tokenizer data/tokenizer_v2.json \
  --reward-modules format accuracy tool ponder \
  --learning-rate 1e-6 \
  --num-rollouts 4 \
  --out-dir checkpoints/rl
```
* **Lưu ý**: Learning rate của RL cực kỳ nhỏ (`1e-6`) để bảo toàn độ ổn định của mạng nơ-ron recurrent core.
* **Đầu ra**: Checkpoint cuối cùng `ckpt_rl_final.pt` đại diện cho trí thông minh System 2 hoàn thiện của Bigram V2!
