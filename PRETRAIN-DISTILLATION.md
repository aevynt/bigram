# ĐẶC TẢ TỰ ĐỘNG CHƯNG CẤT DỮ LIỆU PRETRAIN QUA ĐÊM (PRETRAIN DISTILLATION PIPELINE SPECIFICATION)
*Dành cho AI Agent thực thi cào và chưng cất dữ liệu sách giáo khoa/tri thức chuẩn bị cho Pretrain Bigram V2.*

---

## 🎯 1. MỤC TIÊU & QUY MÔ (TARGET & SCALE)
* **Mục tiêu**: Tạo ra tập dữ liệu chưng cất chất lượng cao dạng sách giáo khoa (textbook), toán logic, giải thuật và code sạch bằng tiếng Việt.
* **Quy mô thí điểm (Pilot Run)**: **100 Triệu đến 1 Tỷ tokens** (khoảng 50,000 đến 500,000 văn bản dài).
* **Quy mô mục tiêu lớn (Full Scale)**: **20 Tỷ đến 40 Tỷ tokens** chạy qua Batch API của **OpenAI (GPT-4o-mini)** và **Anthropic (Claude 3.5 Haiku)**.
* **Hình thức thực thi**: AI Agent chạy tự động không đồng bộ qua đêm (chạy ngầm, gọi Batch API giảm 50% chi phí).

---

## 📂 2. THƯ VIỆN PROMPTS SINH SÁCH GIÁO KHOA CHUYÊN BIỆT (DISTILLATION PROMPT LIBRARY)

AI Agent sử dụng các System Prompts sau để đa dạng hóa chủ đề sinh dữ liệu Pretrain:

### 2.1. Chủ đề 1: Sách giáo khoa Toán học & Logic (Math & Logic Textbook)
```markdown
BẠN LÀ MỘT GIÁO SƯ TOÁN HỌC VÀ LOGIC HỌC ĐẦU NGÀNH.
Nhiệm vụ: Viết một chương sách giáo khoa chi tiết bằng tiếng Việt giảng giải về một chủ đề toán học hoặc logic cụ thể (ví dụ: Lý thuyết tập hợp, Tổ hợp, Xác suất, Đại số tuyến tính, Logic mệnh đề, Hình học giải tích).
Yêu cầu nội dung bắt buộc:
1. Định nghĩa chi tiết, toán học chuẩn xác, giải thích trực quan bằng tiếng Việt chuẩn khoa học.
2. Đưa ra ít nhất 3 ví dụ minh họa từ cơ bản đến nâng cao.
3. Cung cấp bài tập tự luyện kèm lời giải chi tiết từng bước (CoT) sử dụng các suy luận logic chặt chẽ.
4. Định dạng các công thức toán học rõ ràng.
Tuyệt đối không viết lan man, không sử dụng tiếng Anh bồi. Hãy viết dài, chi tiết và học thuật.
```

### 2.2. Chủ đề 2: Sách giáo khoa Khoa học Máy tính & Giải thuật (CS & Algorithms)
```markdown
BẠN LÀ MỘT NHÀ KHOA HỌC MÁY TÍNH KIỆT XUẤT.
Nhiệm vụ: Viết một bài viết học thuật chuyên sâu bằng tiếng Việt về cấu trúc dữ liệu hoặc giải thuật (ví dụ: Cây phân đoạn - Segment Tree, Quy hoạch động, Đồ thị luồng cực đại, Thuật toán so khớp chuỗi, Cơ chế đồng thuận Paxos/Raft).
Yêu cầu nội dung bắt buộc:
1. Giải thích tường tận nguyên lý hoạt động và bản chất toán học của thuật toán.
2. Phân tích độ phức tạp thời gian O(N) và không gian ở các trường hợp tốt nhất, trung bình, xấu nhất.
3. Viết mã nguồn minh họa sạch, tối ưu bằng một trong các ngôn ngữ: C++, Python, Rust hoặc Go (có comment chi tiết từng dòng).
4. Phân tích các edge cases và cách gỡ lỗi thực tế.
```

### 2.3. Chủ đề 3: Sách giáo khoa Vật lý & Khoa học Tự nhiên (Physics & Natural Sciences)
```markdown
BẠN LÀ MỘT NHÀ VẬT LÝ HỌC VÀ KHOA HỌC TỰ NHIÊN ĐỈNH CAO.
Nhiệm vụ: Viết một chương tài liệu khoa học tiếng Việt giảng giải sâu sắc về một hiện tượng lý thuyết hoặc thực nghiệm (ví dụ: Cơ học lượng tử sơ cấp, Điện từ trường Maxwell, Nhiệt động lực học, Thuyết tương đối hẹp, Quang học vật lý).
Yêu cầu nội dung bắt buộc:
1. Mô tả hiện tượng vật lý, thiết lập các phương trình toán học mô tả hiện tượng đó.
2. Giải thích ý nghĩa vật lý của các hằng số và biến số trong phương trình.
3. Đưa ra các ứng dụng thực tiễn trong công nghệ hiện đại.
4. Trình bày bài tập vật lý ứng dụng kèm lời giải chi tiết từng bước.
```

---

## ⚙️ 3. KỊCH BẢN CHẠY BATCH API TỰ ĐỘNG QUA ĐÊM (OVERNIGHT PIPELINE PIPELINE)

AI Agent cần thực hiện quy trình 4 bước tự động dưới đây thông qua script Python `scripts/distill_pretrain_batch.py`:

```
   [Tạo Request List]
           │
           ▼
[Gửi OpenAI/Claude Batch] ──► [Lưu Job ID vào File]
           │
     (Chạy qua đêm)
           │
           ▼
    [Script Polling] ──────► [Tải file kết quả về]
                                    │
                                    ▼
                          [Hậu xử lý & Dọn dẹp] ──► [Nén thành file .bin]
```

### Bước 1: Khởi tạo danh sách yêu cầu (Generator)
Viết script sinh ra file `pretrain_requests.jsonl` chứa hàng ngàn câu hỏi đa dạng dựa trên danh sách các từ khóa khoa học (keywords seed list).

### Bước 2: Submit Batch Job lên OpenAI GPT-4o-mini
```python
# scripts/distill_pretrain_batch.py
import json
import time
from openai import OpenAI

client = OpenAI()

# 1. Tạo file dữ liệu yêu cầu gửi OpenAI
requests = []
topics = ["math", "cs", "physics"]
# Ví dụ tạo 50,000 requests ngẫu nhiên từ bộ từ khóa để chạy qua đêm
for i in range(10000):  
    topic = topics[i % len(topics)]
    system_prompt = ""
    if topic == "math":
        system_prompt = "Hệ thống prompt Giáo sư Toán..."
    elif topic == "cs":
        system_prompt = "Hệ thống prompt Khoa học máy tính..."
    else:
        system_prompt = "Hệ thống prompt Vật lý..."

    requests.append({
        "custom_id": f"pretrain_{topic}_{i:05d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hãy viết chi tiết chương sách về chủ đề số hiệu {i}."}
            ]
        }
    })

# Ghi file jsonl cục bộ
with open("data/openai_pretrain_requests.jsonl", "w", encoding="utf-8") as f:
    for req in requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

# 2. Upload file lên OpenAI
print("Đang tải file request lên OpenAI...")
uploaded_file = client.files.create(
    file=open("data/openai_pretrain_requests.jsonl", "rb"),
    purpose="batch"
)

# 3. Tạo Batch Job
print("Đang khởi chạy OpenAI Batch Job...")
batch_job = client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 4. Ghi lại Job ID để theo dõi qua đêm
with open("data/active_batch_jobs.txt", "a") as log_file:
    log_file.write(f"{batch_job.id}\n")
print(f"Đã gửi Batch thành công! Job ID: {batch_job.id}")
```

### Bước 3: Polling & Tải dữ liệu tự động (Cron/Loop Monitor)
Viết script `scripts/monitor_and_download.py` chạy ngầm để liên tục kiểm tra trạng thái của các Job ID. Khi Job chuyển sang trạng thái `completed`, tải file kết quả về:

```python
# scripts/monitor_and_download.py
import time
from openai import OpenAI

client = OpenAI()

def check_and_download():
    # Đọc danh sách Job ID cần kiểm tra
    with open("data/active_batch_jobs.txt", "r") as f:
        job_ids = [line.strip() for line in f if line.strip()]
    
    remaining_jobs = []
    for job_id in job_ids:
        job = client.batches.retrieve(job_id)
        print(f"Kiểm tra Job {job_id}: Trạng thái = {job.status}")
        
        if job.status == "completed":
            print(f"Job {job_id} đã hoàn thành! Đang tải kết quả...")
            # Tải file kết quả
            output_file_id = job.output_file_id
            file_content = client.files.content(output_file_id).text
            
            # Ghi ra thư mục dữ liệu thô
            output_path = f"data/raw_distilled_{job_id}.jsonl"
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(file_content)
            print(f"Đã lưu kết quả về: {output_path}")
        else:
            remaining_jobs.append(job_id)
            
    # Cập nhật lại danh sách các Job chưa xong
    with open("data/active_batch_jobs.txt", "w") as f:
        for job_id in remaining_jobs:
            f.write(f"{job_id}\n")

if __name__ == "__main__":
    while True:
        check_and_download()
        # Đợi 10 phút trước lần quét tiếp theo
        time.sleep(600)
```

---

## 🧹 4. HẬU XỬ LÝ & NÉN NHỊ PHÂN DỮ LIỆU (POST-PROCESSING & BIN PACK)

Sau khi tải dữ liệu về, AI Agent phải lọc sạch tạp chất và nén nhị phân:

1. **Lọc Regex & AST**: Loại bỏ hoàn toàn các câu thừa (ví dụ: *"Dưới đây là bài viết..."*), chuẩn hóa NFC bằng thư viện `unicodedata.normalize('NFC', text)`.
2. **Trích xuất nội dung**: Lấy phần text thô trong kết quả trả về của GPT-4o-mini ghép lại thành một file văn bản lớn `data/distilled_pretrain.txt`.
3. **Nén nhị phân (Data Bin Pack)**: Chạy script `prepare_data.py` (đã có trong codebase) sử dụng Tokenizer **VS-BPE** để mã hóa và nén toàn bộ văn bản thô thành định dạng `.bin` và `.idx` sẵn sàng đưa thẳng vào GPU huấn luyện Pretrain.

---

## 📋 YÊU CẦU GIAO VIỆC CHO AI AGENT CHẠY QUA ĐÊM (HANDOFF INSTRUCTION TO AGENT)
> *Dán trực tiếp câu lệnh này vào cửa sổ chat của AI Agent thực thi:*
>
> "Chào Agent, hãy đọc tài liệu đặc tả [PRETRAIN-DISTILLATION.md](file:///c:/Users/lhqua/Documents/bigram/PRETRAIN-DISTILLATION.md). Nhiệm vụ của bạn đêm nay là thiết lập toàn bộ pipeline chưng cất dữ liệu sách giáo khoa pretrain tiếng Việt bằng OpenAI GPT-4o-mini Batch API. Hãy viết các script `scripts/distill_pretrain_batch.py` và `scripts/monitor_and_download.py`, chạy thử nghiệm pilot 100 requests trước để kiểm tra lỗi, sau đó kích hoạt submit Job Batch lớn 50,000 requests. Chạy ngầm tiến trình giám sát và tự động tải kết quả về thư mục `data/` khi hoàn thành. Hãy báo cáo trạng thái Job ID trước khi tôi đi ngủ."
