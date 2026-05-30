# ĐẶC TẢ CHƯNG CẤT DỮ LIỆU TRỰC TIẾP BẰNG AI AGENT (DIRECT AGENT DISTILLATION BLUEPRINT)
*Đặc tả kỹ thuật yêu cầu AI Agent tự sinh và ghi trực tiếp dữ liệu học tập cao cấp vào thư mục `distilldata/`.*

Tài liệu này đóng vai trò là bản hướng dẫn hành động (Actionable Handoff Prompt) để bạn giao việc trực tiếp cho một AI Agent có khả năng lập trình/ghi file. AI Agent sẽ tự đóng vai trò là bộ máy chưng cất tri thức, tự sinh nội dung học thuật tiếng Việt chất lượng cao và ghi trực tiếp thành các tệp tin JSONL sạch tại thư mục `distilldata/` bằng công cụ viết file của mình mà không cần gọi qua API hay chạy code Python phụ trợ.

---

## 📂 1. CẤU TRÚC THƯ MỤC & ĐỊNH DẠNG DỮ LIỆU ĐẦU RA

AI Agent phải tạo thư mục `distilldata/` ở thư mục gốc của dự án (nếu chưa tồn tại) và phân bổ các tệp tin dữ liệu theo định dạng **JSON Lines (JSONL)**:

```
bigram/
├── distilldata/
│   ├── pretrain_math_001.jsonl      <-- Dữ liệu sách giáo khoa Toán & Logic
│   ├── pretrain_cs_001.jsonl        <-- Dữ liệu Khoa học Máy tính & Giải thuật
│   └── pretrain_physics_001.jsonl   <-- Dữ liệu Vật lý & Khoa học Tự nhiên
```

### Định dạng từng dòng (JSON Line Schema):
Mỗi dòng phải là một đối tượng JSON chuẩn hóa UTF-8 NFC, chứa cấu trúc:
```json
{"prompt": "Yêu cầu học tập/Câu hỏi chuyên sâu", "response": "<think>\nSuy nghĩ lập luận đa bước của AI\n</think>\nNội dung chi tiết sách giáo khoa/Code/Ví dụ minh họa chi tiết..."}
```

---

## 🎯 2. TIÊU CHUẨN NỘI DUNG THEO TỪNG CHỦ ĐỀ

AI Agent phải tuân thủ nghiêm ngặt các tiêu chuẩn chất lượng khi tự sinh nội dung:

### 2.1. Chủ đề 1: Sách giáo khoa Toán học & Logic (`pretrain_math_*.jsonl`)
* **Nội dung**: Lý thuyết tập hợp, tổ hợp, xác suất, hình học giải tích, đại số tuyến tính, logic mệnh đề.
* **Yêu cầu**: 
  - Giải thích định nghĩa toán học rõ ràng bằng tiếng Việt học thuật chuẩn xác.
  - Phải có phần lập luận logic rõ ràng từng bước trong thẻ `<think>`.
  - Đưa ra tối thiểu 3 ví dụ trực quan kèm bài tập tự luyện và lời giải chi tiết.

### 2.2. Chủ đề 2: Sách giáo khoa Khoa học Máy tính & Giải thuật (`pretrain_cs_*.jsonl`)
* **Nội dung**: Quy hoạch động, cấu trúc dữ liệu nâng cao (Segment Tree, Fenwick), giải thuật đồ thị, hệ thống phân tán, cơ chế đồng thuận.
* **Yêu cầu**:
  - Giải thích nguyên lý hoạt động và phân tích độ phức tạp thời gian/không gian $O(N)$.
  - Viết code minh họa sạch, tối ưu (C++, Python, Rust hoặc Go) kèm giải thích chi tiết.

### 2.3. Chủ đề 3: Sách giáo khoa Vật lý & Khoa học Tự nhiên (`pretrain_physics_*.jsonl`)
* **Nội dung**: Cơ học lượng tử, điện từ trường, thuyết tương đối, quang học, hóa lý.
* **Yêu cầu**:
  - Thiết lập phương trình toán học mô tả hiện tượng vật lý.
  - Giải thích tường tận ý nghĩa vật lý của các hằng số/biến số.
  - Đưa ra bài tập thực tiễn kèm lời giải chi tiết.

---

## 📋 YÊU CẦU GIAO VIỆC CHO AI AGENT THỰC THI (HANDOFF PROMPT TO AGENT)
> *Sao chép toàn bộ nội dung dưới đây và dán trực tiếp vào khung chat của AI Agent có quyền ghi file để bắt đầu quá trình sinh dữ liệu tự động:*

```markdown
Nhiệm vụ của bạn là hoạt động như một Bộ Máy Chưng Cất Dữ Liệu Học Máy (ML Data Distillation Engine). Bạn sẽ tự suy nghĩ và sinh ra kho dữ liệu học thuật tiếng Việt cao cấp, sau đó sử dụng công cụ ghi file của mình để tạo trực tiếp các file JSONL sạch vào thư mục `distilldata/` của dự án.

QUY TẮC THỰC THI BẮT BUỘC:
1. Hãy tạo thư mục `distilldata/` tại thư mục gốc của dự án (nếu chưa có).
2. Hãy sinh trực tiếp dữ liệu học tập cao cấp cho 3 nhóm chủ đề: Toán học (Math), Khoa học máy tính (CS), và Vật lý (Physics).
3. Ghi dữ liệu vào các file tương ứng:
   - `distilldata/pretrain_math_001.jsonl` (Ít nhất 50 mẫu chuyên sâu về Toán/Logic)
   - `distilldata/pretrain_cs_001.jsonl` (Ít nhất 50 mẫu chuyên sâu về Giải thuật/Code)
   - `distilldata/pretrain_physics_001.jsonl` (Ít nhất 50 mẫu chuyên sâu về Vật lý/Khoa học tự nhiên)
4. Mỗi mẫu phải là một dòng JSONL chuẩn:
   {"prompt": "Câu hỏi/Yêu cầu học tập", "response": "<think>\nLập luận logic\n</think>\nNội dung bài học/Code/Lời giải chi tiết"}
5. Tuyệt đối không viết code chạy Python để gọi API hay gửi Batch API nào cả. Nhiệm vụ của bạn là TỰ SUY NGHĨ, SINH NỘI DUNG VÀ GHI TRỰC TIẾP thành các tệp tin dữ liệu thô này vào thư mục `distilldata/` bằng công cụ ghi file của mình.
6. Đảm bảo ngôn ngữ tiếng Việt tự nhiên, chuẩn NFC, học thuật, không pha tạp tiếng Anh bồi. Các đoạn code minh họa phải tối ưu và được định dạng markdown chuẩn trong chuỗi JSON.

Hãy tiến hành sinh dữ liệu chi tiết và ghi trực tiếp vào các tệp tin trên ngay bây giờ!
```
