# ĐẶC TẢ CHƯNG CẤT DỮ LIỆU SFT (SFT-DATA DISTILLATION BLUEPRINT) - CLAUDE & OPENAI GPT
*Bộ Prompt và quy trình dành cho AI Agent để tự động cào và chưng cất dữ liệu SFT/Tool calling cao cấp từ Anthropic Claude và OpenAI GPT.*

Tài liệu này đặc tả cấu trúc dữ liệu SFT mong muốn và cung cấp các **System Prompts đỉnh cao** giúp AI Agent tự động gọi Batch API từ cả **Anthropic Claude** và **OpenAI GPT (GPT-4o / GPT-4o-mini)** để chưng cất hàng triệu dòng dữ liệu sạch tuyệt đối.

---

## 📂 1. ĐẶC TẢ ĐỊNH DẠNG DỮ LIỆU ĐẦU RA (SFT DATA SCHEMA)

Mọi tệp dữ liệu SFT chưng cất phải được xuất ra dưới dạng **JSONL**, mã hóa UTF-8 sạch, chuẩn hóa Unicode NFC. 

### Định dạng hội thoại chuẩn tiếng Việt:
```json
{
  "prompt": "Xin chào, bạn có thể giúp tôi giải thích bản chất của thuật toán Dijkstra không?",
  "response": "<think>\nNgười dùng muốn biết về thuật toán Dijkstra.\n1. Định nghĩa: Tìm đường đi ngắn nhất từ 1 đỉnh nguồn.\n2. Nguyên lý: Tham lam (Greedy), sử dụng hàng đợi ưu tiên (Priority Queue).\n3. Trình bày: Rõ ràng, trực quan bằng tiếng Việt chuẩn khoa học.\n</think>\nThuật toán Dijkstra là một thuật toán đồ thị kinh điển được sử dụng để tìm đường đi ngắn nhất..."
}
```

---

## 🎯 2. BỘ PROMPT CHƯNG CẤT CHUYÊN BIỆT (DISTILLATION PROMPTS)

AI Agent sẽ nạp các System Prompt sau đây tương ứng với từng nhóm dữ liệu khi gọi API:

### 2.1. Nhóm 1: Suy luận đa bước chuyên sâu (System 2 CoT Reasoning)
* **Mục tiêu**: Huấn luyện mô hình cách tư duy logic từng bước, tự sửa lỗi trước khi trả lời.
* **System Prompt dành cho Claude & GPT-4o**:
```markdown
BẠN LÀ MỘT CHUYÊN GIA TOÁN HỌC VÀ LOGIC HỌC HÀNG ĐẦU.
Nhiệm vụ: Hãy tạo ra một câu hỏi toán đố, logic hoặc giải thuật hóc búa bằng tiếng Việt, sau đó viết câu trả lời hoàn chỉnh.
Quy tắc phản hồi bắt buộc:
1. Câu trả lời của bạn BẮT BUỘC phải mở đầu bằng thẻ `<think>`.
2. Bên trong thẻ `<think>`, hãy viết toàn bộ chuỗi suy luận logic, phân tích giả thuyết, các phép thử sai và quá trình tự sửa lỗi của bạn bằng tiếng Việt chuẩn.
3. Đóng thẻ bằng `</think>`, sau đó mới viết lời giải chính xác, tường tận và dễ hiểu cho người dùng.
4. Tuyệt đối không pha trộn từ tiếng Anh bồi.
```

### 2.2. Nhóm 2: Stateful Tool Calling (Gọi công cụ có trạng thái)
* **Mục tiêu**: Huấn luyện mô hình cú pháp gọi tool hoàn hảo và cách phản hồi dựa trên kết quả trả về của tool.
* **System Prompt dành cho Claude & GPT-4o**:
```markdown
BẠN LÀ SIÊU TÁC TỬ HỆ THỐNG (SYSTEM AGENT).
Nhiệm vụ: Tạo ra một kịch bản gọi công cụ hoàn chỉnh để giải quyết một yêu cầu thực tế của người dùng.
Các công cụ có sẵn trong Registry của chúng tôi gồm:
  - `calculator` (đối số: `expression`)
  - `python.run` (đối số: `code`)
  - `terminal.run` (đối số: `command`)
  - `code.search` (đối số: `query`)

Quy tắc phản hồi bắt buộc:
1. Trả về đúng cú pháp gọi tool trong đoạn hội thoại tự hồi quy:
   `<tool_call> {"name": "tên_công_cụ", "args": {"đối_số": "giá_trị"}} </tool_call>`
2. Sau khi gọi tool, hãy giả lập kết quả trả về của tool trong thẻ:
   `<tool_response> kết_quả_giả_lập </tool_response>`
3. Cuối cùng, đưa ra câu trả lời phân tích dựa trên kết quả đó.
```

### 2.3. Nhóm 3: Lập trình cao cấp và Debug (Advanced Programming)
* **Mục tiêu**: Huấn luyện mô hình khả năng viết code chuẩn xác và giải thích code tường tận.
* **System Prompt dành cho Claude & GPT-4o**:
```markdown
BẠN LÀ MỘT KIẾN TRÚC SƯ PHẦN MỀM THƯỢNG HẠNG.
Nhiệm vụ: Tạo ra một bài toán lập trình giải thuật phức tạp bằng tiếng Việt (ví dụ: tối ưu hóa cache, hệ thống phân tán, xử lý bất đồng bộ trong C++, Python, Rust hoặc Go).
Quy tắc phản hồi bắt buộc:
1. Viết code sạch, tối ưu, có comment giải thích rõ ràng.
2. Viết tài liệu hướng dẫn kỹ thuật chuyên sâu (Whitepaper) phân tích độ phức tạp thời gian O(N) và không gian, các edge cases và cách xử lý lỗi.
3. Định dạng mã nguồn chuẩn trong khối code markdown.
```

---

## ⚙️ 3. QUY TRÌNH CHẠY BATCH API TỰ ĐỘNG (PIPELINE EXECUTION)

Để tối thiểu hóa chi phí, AI Agent phải sử dụng tính năng **Batch API (giảm 50% chi phí)** của cả Anthropic và OpenAI. Dưới đây là đặc tả định dạng gửi yêu cầu.

### 3.1. Phân hệ 1: Anthropic Batch API (Dành cho Claude 3.5 Haiku/Sonnet)
Gom các request thành file `.jsonl` theo cấu trúc:
```json
{
  "custom_id": "math_reasoning_claude_001",
  "params": {
    "model": "claude-3-5-haiku-20241022",
    "max_tokens": 2048,
    "messages": [
      {"role": "user", "content": "Hãy tạo một bài toán suy luận logic phức tạp và giải nó."}
    ],
    "system": "Hệ thống prompt đặc tả tương ứng ở Mục 2..."
  }
}
```

#### Code Python submit Batch Anthropic:
```python
import anthropic
client = anthropic.Anthropic()
batch = client.beta.messages.batches.create(
    requests=[/* Load danh sách request từ file JSONL */]
)
print(f"Đã gửi Anthropic Batch thành công! Job ID: {batch.id}")
```

### 3.2. Phân hệ 2: OpenAI Batch API (Dành cho GPT-4o / GPT-4o-mini)
GPT-4o-mini Batch API siêu rẻ (**$0.30 / triệu output tokens**), rất thích hợp để chưng cất hàng chục tỷ token tri thức nền tảng.
Gom các request thành file `.jsonl` theo cấu trúc:
```json
{
  "custom_id": "math_reasoning_gpt_001",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o-mini",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "Hệ thống prompt đặc tả tương ứng ở Mục 2..."},
      {"role": "user", "content": "Hãy tạo một bài toán suy luận logic phức tạp và giải nó."}
    ]
  }
}
```

#### Code Python submit Batch OpenAI:
```python
from openai import OpenAI
client = OpenAI()

# 1. Tải file request lên OpenAI
file_upload = client.files.create(
  file=open("openai_batch_requests.jsonl", "rb"),
  purpose="batch"
)

# 2. Khởi động Batch Job
batch = client.batches.create(
  input_file_id=file_upload.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
print(f"Đã gửi OpenAI Batch thành công! Job ID: {batch.id}")
```
AI Agent giám sát trạng thái Job ID của cả hai bên qua đêm, tải kết quả về, lọc sạch tạp chất qua màng lọc Regex/AST và lưu trữ trực tiếp vào thư mục `data/` của dự án Bigram.
