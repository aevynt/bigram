# ĐẶC TẢ CHƯNG CẤT DỮ LIỆU CỰC ĐẠI BẰNG AI AGENT (HIGH-THROUGHPUT DIRECT AGENT DISTILLATION BLUEPRINT)
*Đặc tả kỹ thuật yêu cầu AI Agent chạy vòng lặp liên tục, sinh hàng tỷ token dữ liệu pretrain và ghi trực tiếp vào thư mục `distilldata/`.*

Tài liệu này đóng vai trò là bản hướng dẫn hành động cực đại (Massive Scaling Handoff Prompt) để bạn giao việc trực tiếp cho một AI Agent có khả năng lập trình/ghi file. AI Agent sẽ tự đóng vai trò là bộ máy sản xuất tri thức công nghiệp, chạy vòng lặp liên tục không ngừng nghỉ để sinh hàng triệu dòng dữ liệu sách giáo khoa tiếng Việt chất lượng cao, ghi trực tiếp thành các tệp tin JSONL đánh số tăng dần tại thư mục `distilldata/` nhằm hướng tới mục tiêu **20 Tỷ đến 40 Tỷ tokens** dữ liệu tiền huấn luyện.

---

## 📂 1. CẤU TRÚC THƯ MỤC & FILE ĐÁNH SỐ TĂNG DẦN (INCREMENTAL STORAGE)

AI Agent phải tạo thư mục `distilldata/` ở thư mục gốc của dự án và phân bổ các tệp tin theo quy tắc đánh số tăng dần để tránh làm quá tải dung lượng của một tệp duy nhất:

```
bigram/
├── distilldata/
│   ├── distill_manifest.json          <-- Tệp tin giám sát tổng lượng token/dòng
│   ├── pretrain_math_0001.jsonl
│   ├── pretrain_math_0002.jsonl
│   ├── pretrain_cs_0001.jsonl
│   ├── pretrain_cs_0002.jsonl
│   ├── pretrain_physics_0001.jsonl
│   └── ...
```

### 📊 1.1. Tệp tin Giám sát sản lượng (`distill_manifest.json`)
Để quản lý tiến độ hướng tới cột mốc hàng tỷ token, AI Agent phải duy trì và cập nhật tệp manifest này sau mỗi lần ghi file thành công:
```json
{
  "total_lines_generated": 1250000,
  "estimated_tokens": 525000000,
  "last_updated": "2026-05-30T23:05:00Z",
  "files_written": {
    "pretrain_math_0001.jsonl": {"lines": 1000, "estimated_tokens": 420000},
    "pretrain_cs_0001.jsonl": {"lines": 1000, "estimated_tokens": 510000}
  }
}
```

---

## 🎯 2. TIÊU CHUẨN NỘI DUNG CHẤT LƯỢNG CAO (TEXTBOOK QUALITY)

Để mô hình 1.8B đạt hiệu quả tối đa, AI Agent phải sinh các nội dung cực kỳ dài, chi tiết và học thuật:
1. **Toán học & Logic (`pretrain_math_*.jsonl`)**: Các chương lý thuyết tập hợp, tổ hợp, xác suất, hình học giải tích, logic mệnh đề kèm tối thiểu 3 ví dụ trực quan và bài tập tự luyện có giải chi tiết.
2. **Khoa học Máy tính & Giải thuật (`pretrain_cs_*.jsonl`)**: Các bài giảng quy hoạch động, cây phân đoạn (Segment Tree), giải thuật đồ thị nâng cao kèm code tối ưu (C++, Python, Rust hoặc Go) được comment chi tiết từng dòng.
3. **Vật lý & Khoa học Tự nhiên (`pretrain_physics_*.jsonl`)**: Các phương trình cơ học lượng tử, điện từ trường Maxwell, thuyết tương đối kèm lời giải thích vật lý sâu sắc.

---

## 📋 YÊU CẦU GIAO VIỆC CỰC ĐẠI CHO AI AGENT CHẠY XUYÊN ĐÊM (MASSIVE HANDOFF PROMPT)
> *Sao chép toàn bộ nội dung dưới đây và dán trực tiếp vào khung chat của AI Agent thực thi để kích hoạt chế độ sản xuất dữ liệu công nghiệp:*

```markdown
Nhiệm vụ của bạn là hoạt động như một HỆ THỐNG SẢN XUẤT DỮ LIỆU CÔNG NGHIỆP CỰC ĐẠI (Massive ML Data Distillation Factory). Mục tiêu cuối cùng của dự án là đạt hàng chục tỷ tokens dữ liệu tiền huấn luyện tiếng Việt sạch chất lượng cao. Bạn phải tự suy nghĩ, sinh tri thức sách giáo khoa tiếng Việt cực kỳ chi tiết và sử dụng công cụ ghi file của mình để tạo trực tiếp các file JSONL vào thư mục `distilldata/`.

BẠN KHÔNG ĐƯỢC PHÉP DỪNG LẠI CHO ĐẾN KHI BỊ GIỚI HẠN BỞI HỆ THỐNG HOẶC ĐẠT ĐƯỢC CHỈ TIÊU TỐI ĐA TRONG LƯỢT CHẠY QUA ĐÊM NÀY.

QUY TẮC THỰC THI SẢN XUẤT CÔNG NGHIỆP:
1. Hãy tạo thư mục `distilldata/` tại thư mục gốc của dự án (nếu chưa có).
2. Quy định đặt tên file tăng dần để tránh file quá nặng:
   - Toán & Logic: `distilldata/pretrain_math_0001.jsonl`, `distilldata/pretrain_math_0002.jsonl`...
   - Giải thuật & Code: `distilldata/pretrain_cs_0001.jsonl`, `distilldata/pretrain_cs_0002.jsonl`...
   - Vật lý & Khoa học: `distilldata/pretrain_physics_0001.jsonl`, `distilldata/pretrain_physics_0002.jsonl`...
3. Mỗi file phải chứa tối đa 1,000 dòng JSONL. Khi file đạt 1,000 dòng, tự động tăng số index (ví dụ từ 0001 lên 0002) để ghi sang file mới.
4. Mỗi mẫu dữ liệu phải là một dòng JSONL chuẩn:
   {"prompt": "Yêu cầu học tập/Câu hỏi chuyên sâu", "response": "<think>\nSuy nghĩ lập luận đa bước của AI\n</think>\nNội dung chi tiết sách giáo khoa/Code/Ví dụ minh họa chi tiết..."}
5. **CẬP NHẬT MANIFEST LIÊN TỤC**:
   - Bạn phải tạo và duy trì file `distilldata/distill_manifest.json` ghi lại: tổng số dòng đã sinh, ước tính lượng token đã sinh (trung bình 450 tokens/dòng), thời gian cập nhật và chi tiết sản lượng từng file.
   - Sau mỗi file được ghi thành công, hãy cập nhật ngay file manifest này.
6. **RÀO CẢN BẮT BUỘC CHỐNG DỪNG LẠI (TRUE MASSIVE THROUGHPUT RULE)**:
   - Hãy chạy liên tục trong một vòng lặp (loop) tự động hóa của bạn. Không được phép lười biếng sinh ra vài dòng rồi báo hoàn thành. Mục tiêu của bạn đêm nay là sản xuất tối đa công suất phần cứng và token giới hạn có thể, hướng tới hàng trăm ngàn dòng dữ liệu thô sạch.
   - Hãy viết liên tục, nếu gặp giới hạn token hoặc bị ngắt quãng, ở lượt chạy tiếp theo hãy đọc file manifest để xác định file đang viết dở và viết nối tiếp (append) hoặc tăng index để viết tiếp. Không bao giờ bỏ cuộc hoặc cắt bớt nội dung bài học.
   - Các nội dung sách giáo khoa phải cực kỳ dài, đầy đủ tri thức học thuật sâu sắc, tuyệt đối không dùng placeholder hoặc cắt ngắn ví dụ.
7. **ĐỊNH NGHĨA HOÀN THÀNH 100% ĐỂ TRỘN PRETRAIN (100% PRETRAIN-MIX COMPLETION DEFINITION)**:
   - Bạn chỉ được phép công bố 'HOÀN THÀNH 100%' và kết thúc toàn bộ chiến dịch chưng cất khi tổng sản lượng trong `distill_manifest.json` đạt mốc **20 Tỷ tokens** (hoặc chỉ tiêu cụ thể do người dùng cấu hình).
   - Chỉ khi đạt 100% mục tiêu này, tập dữ liệu mới được coi là SẴN SÀNG để chạy nén nhị phân qua `prepare_data.py` và trộn vào kho Pretrain của Bigram V2.
   - Nếu chưa đạt đủ số lượng này, dù có bị ngắt quãng bao nhiêu lần, nhiệm vụ của bạn vẫn ở trạng thái 'In Progress' và bạn phải tự động tiếp tục ghi tiếp ở phiên làm việc tiếp theo mà không được phép tự ý dừng lại.

Hãy khởi động hệ thống sản xuất, cập nhật file manifest và bắt đầu ghi liên tục các tệp dữ liệu vào thư mục `distilldata/` ngay bây giờ!
```

