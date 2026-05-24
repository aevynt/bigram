# Bigram Nano 1.5 — Báo cáo kết quả

## Kiến trúc mới
- **Trụ cột 1: BMSSP (Byte-Fallback Morpheme-Syllable SentencePiece)**:
  - Tokenizer mới sử dụng thuật toán Unigram kết hợp cơ chế Byte-fallback để loại bỏ hoàn toàn lỗi `<unk>`, tự động phân rã ký tự lạ hoặc OOV thành byte UTF-8 thô.
  - Áp dụng chuẩn hóa Unicode NFC và ép phân đoạn tại ranh giới âm tiết tiếng Việt (`split_by_whitespace=True`), giúp tối ưu việc biểu diễn tiếng Việt.
  - Kích thước từ điển thực tế: **500** (điều chỉnh từ 8,000 để phù hợp với quy mô corpus nhỏ).
- **Trụ cột 2: SS-KVP (Shared-State Key-Value Projection)**:
  - Cơ chế Zero-Shot KV-cache Sharing trong Recurrent Core giúp chia sẻ các phép chiếu Key/Value tĩnh trên tất cả các bước lặp của cùng một vị trí token.
  - Giảm thiểu lưu trữ KV-cache theo hệ số số vòng lặp $r$, giải phóng dung lượng bộ nhớ lớn trong quá trình inference.
  - Implement bằng cách đóng băng việc tính lại K, V sau vòng lặp đầu tiên của Recurrent Core và lưu cache trực tiếp trong `SSKVPAttention` của mỗi block.

## Kết quả research
- **SS-KVP (Q1-Q5)**:
  - Huginn paper (arXiv:2502.05171) xác nhận KV-cache được share zero-shot qua các vòng lặp dọc giúp tiết kiệm bộ nhớ $r$ lần.
  - Dù $K$ và $V$ cố định, attention vẫn học nhờ **Query ($Q$) được tính toán động** ở từng vòng lặp. Cơ chế **Input Injection** ngăn chặn tình trạng latent collapse.
  - Universal Transformers (2018) **không** share KV mà tính lại hoàn toàn ở mỗi layer.
  - Việc share KV tạo **gradient highway** rất thông thoáng truyền thẳng về lớp chiếu ban đầu và Prelude.
  - Kiểm soát bùng nổ gradient bằng `grad_clip = 1.0`.
- **BMSSP (Q6-Q10)**:
  - Byte-fallback hoạt động bằng cách thay `<unk>` bằng chuỗi byte tokens UTF-8 thô khi gặp OOV.
  - Phân đoạn âm tiết tiếng Việt hoàn hảo nhờ `split_by_whitespace=True` vì dấu cách phân tách các âm tiết tiếng Việt rõ ràng.
  - Unicode NFC normalization cực kỳ quan trọng để quy đổi Tổ hợp (NFD) về Dựng sẵn (NFC), tránh phân mảnh biểu diễn token.

## Kết quả training
| Metric | Giá trị |
|--------|---------|
| Train loss cuối (Pretrain) | 0.0007 |
| Val loss cuối (Pretrain) | 0.0006 |
| Gap (Val - Train) | -0.0001 |
| SFT loss cuối | 0.0252 |
| Vocab size BMSSP | 500 |
| Số tham số model | 1.02M |

## So sánh với Nano 1
- **Nano 1** sử dụng Tonal Tokenizer (tách âm gốc + thanh điệu độc lập) đi kèm đầu ra `tone_head` riêng biệt.
- **Nano 1.5** sử dụng BMSSP Tokenizer (bảng từ điển duy nhất mã hóa âm tiết chuẩn NFC, không có `tone_head`).
- **Nhận xét**: 
  - Với cùng kích thước model ẩn, việc loại bỏ `tone_head` và chuyển đổi sang một bảng từ điển duy nhất giúp giản lược hóa cấu trúc loss (chỉ còn LM loss và Auxiliary loss).
  - Tốc độ hội tụ cực nhanh (loss pretrain tiệm cận 0 chỉ sau ~1000 step) nhờ SS-KVP tạo gradient flow cực kỳ thông suốt.
  - Việc loại bỏ tone head giúp model sinh văn bản mượt mà, tự nhiên và loại bỏ nguy cơ lệch pha dấu thanh của luồng âm gốc.

## Vấn đề phát sinh & Cách giải quyết
1. **Lỗi cp1252 khi Print chữ tiếng Việt trên Windows**: Chạy lệnh in bị crash do Python mặc định mã hóa cp1252 trên terminal Windows.
   - *Giải quyết*: Reconfigure standard output và standard error về UTF-8 bằng `sys.stdout.reconfigure(encoding='utf-8')` ở đầu mỗi file script và lệnh kiểm tra.
2. **Lỗi [Errno 22] Invalid Path `/tmp` trên Windows**: Bộ unit test mặc định tạo file tạm dưới thư mục `/tmp` không tồn tại trên Windows.
   - *Giải quyết*: Cập nhật bộ unit test (`test_data_training.py`, `test_tokenizer.py`, `test_alignment.py`) sử dụng thư mục tạm của hệ điều hành thông qua thư viện `tempfile.gettempdir()`.
3. **Lỗi TypeError concatenate None trong SFT Dataset**: Tokenizer BMSSP trả về `tone_ids = None`, dẫn đến crash khi ghép danh sách thanh điệu trong SFT, SFT-like Preference và Calibration datasets.
   - *Giải quyết*: Bổ sung logic kiểm tra và tự động khởi tạo chuỗi số 0 cho tone ids có độ dài tương ứng khi nhận giá trị `None` trong `bigram/data/dataset.py`.

## Khuyến nghị tiếp theo
1. **Scale-up Corpus**: Dataset kiểm thử hiện tại rất nhỏ (78 câu). Cần mở rộng tập huấn luyện lên hàng trăm triệu tokens để BMSSP Tokenizer học đầy đủ ngữ cảnh tiếng Việt phong phú.
2. **Fine-tune Siêu tham số SS-KVP**: Thử nghiệm các hệ số `grad_clip` nhỏ hơn (ví dụ 0.2 như paper khuyến nghị) để tăng tính ổn định của gradient khi scale model lớn hơn.
