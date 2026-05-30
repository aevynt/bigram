# THẺ MÔ HÌNH (MODEL CARD): BIGRAM-TENSOR-1.8B (V2)
*Compact Vietnamese-First Recurrent-Depth Hybrid-MoE Language Model.*

**Bigram-Tensor-1.8B (V2)** là mô hình ngôn ngữ tiếng Việt thế hệ mới được xây dựng trên siêu kiến trúc lai **Recurrent-depth (Tuần hoàn-Chiều sâu)**, kết hợp nén attention nâng cao (MLA), State Space Models (Mamba-2) và Mixture of Experts (MoE).

---

## 🔬 1. THÔNG SỐ KIẾN TRÚC CHI TIẾT (ARCHITECTURE SPECS)

Mô hình sử dụng thiết kế phân mảnh Prelude-Recurrent-Coda tối ưu hóa chi phí phần cứng:

| Thông số kỹ thuật | Giá trị cấu hình | Mô tả chi tiết |
| :--- | :--- | :--- |
| **Active Parameters** | **1.8 Billion** | Tham số chạy thực tế trên mỗi token. |
| **Total Parameter Capacity**| **3.5 Billion** | Dung lượng tham số thực tế nhờ cấu trúc MoE. |
| **Vocab Size** | **32,000** | Tokenizer đơn luồng **VS-BPE** nhận biết âm tiết tiếng Việt. |
| **Hidden Size ($h$)** | **2048** | Chiều kích thước của vector ẩn residual stream. |
| **Intermediate Size** | **5504** | Chiều ẩn của SwiGLU FFN experts. |
| **Context Length** | **16,384 (16k)** | **Tương đương ~48k tokens Llama-3** nhờ nén âm tiết. |
| **Prelude Layers ($l_P$)** | **2** | Layer đưa input vào không gian latent ẩn. |
| **Recurrent Layers ($l_R$)** | **6** | Lõi tuần hoàn unroll động, đan xen **3 MLA + 3 Mamba-2**. |
| **Coda Layers ($l_C$)** | **2** | Layer giải mã latent state ẩn ra dự đoán. |
| **Attention Mechanism** | **MLA (DeepSeek-V3)**| Nén KV cache ($d_c=128$, Decoupled RoPE $d_R=64$). |
| **State Space Model** | **Mamba-2 (SSD)** | Block SSM thuần PyTorch đóng vai trò long-term memory. |
| **Mixture of Experts** | **Shared-MoE** | 8 fine-grained experts (Top-2 active) + 1 Shared Expert. |
| **Dynamic Halting** | **PonderNet** | Halting head unroll động xác suất khả vi đầy đủ. |

---

## ⚡ 2. CÁC NĂNG LỰC ĐỘT PHÁ SOTA (CORE CAPABILITIES)

### 1. Tương thích 100% Hệ sinh thái Công nghiệp (Ecosystem Native)
Không còn sử dụng cơ chế hai luồng (Dual-stream) phức tạp ở bản gốc, Bigram V2 sử dụng **đơn luồng (Single-Stream) chuẩn hóa**.
* Mô hình có thể deploy trực tiếp lên các engine suy luận siêu tốc như **vLLM (PagedAttention)**, Hugging Face `transformers` và TensorRT-LLM mà không cần sửa đổi bất kỳ dòng code C++/CUDA nào của engine.

### 2. Tư duy sâu sắc System 2 (Implicit Latent Reasoning)
* Nhờ unroll động qua **PonderNet**, mô hình tự động điều tiết compute: "nghĩ lâu" khi gặp toán khó, và dừng sớm với câu hỏi dễ.
* Tích hợp thuật toán **Latent Beam Search** ở test-time, sử dụng `verifier_head` để rẽ nhánh và tìm kiếm quỹ đạo tư duy ẩn tối ưu trước khi phát ngôn.

### 3. Stateful Tool Calling hoàn hảo
* Lớp mạng `tool_head` kết hợp với **Grammar-Constrained Decoding (Outlines)** chuyển đổi JSON Schema của tool trực tiếp thành Finite State Machine (FSM).
* Triệt tiêu hoàn toàn 100% lỗi cú pháp đầu ra của tool call, biến Bigram V2 thành một Agent vận hành terminal, viết code Python và tra cứu RAG cực kỳ an toàn.

---

## 🚀 3. HƯỚNG DẪN KHỞI TẠO BẰNG CODE (MODEL INITIALIZATION)

Bạn có thể dễ dàng khởi tạo mô hình này trực tiếp bằng Python sử dụng codebase hiện tại:

```python
import torch
from bigram.config import bigram_v2_1_8b_a6000_config
from bigram.model.bigram import BigramModel

# 1. Nạp cấu hình vàng tối ưu cho GPU RTX A6000
cfg = bigram_v2_1_8b_a6000_config()

# 2. Khởi tạo mô hình
model = BigramModel(cfg.model)
print(f"Tổng số tham số học được: {model.num_parameters():,}")
# Đầu ra: ~1.8B Active (3.5B Total Capacity)

# 3. Kích hoạt tính năng sinh suy luận ẩn System 2
token_ids = torch.randint(0, 32000, (1, 10))
generated, _, abstained = model.generate_with_latent_reasoning(
    token_ids=token_ids,
    max_new_tokens=50,
    beam_width=3,
    num_recurrence=16
)
```

---

## 📈 4. THÔNG TIN BẢN QUYỀN & PHÁT HÀNH
* **Nhà phát triển**: Cộng đồng nghiên cứu Open-Source Bigram.
* **Giấy phép (License)**: Apache-2.0.
* **Nền tảng huấn luyện**: Windows Server + 1x NVIDIA RTX A6000 48GB.
