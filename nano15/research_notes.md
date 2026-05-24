# Báo cáo Nghiên cứu & Xác minh Kỹ thuật — Bigram Nano 1.5

Tài liệu này ghi nhận kết quả nghiên cứu lý thuyết và xác minh kỹ thuật cho hai trụ cột mới của Bigram Nano 1.5: **BMSSP (Byte-Fallback Morpheme-Syllable SentencePiece)** và **SS-KVP (Shared-State Key-Value Projection)**.

---

## ## RESEARCH 1A: Xác minh SS-KVP từ Huginn paper

### Q1: Trong Huginn, Key và Value có được tính lại ở mỗi vòng lặp không, hay được cache/share từ vòng đầu? Trích dẫn đoạn cụ thể trong paper.
* **Trả lời:** Trong Huginn, Key và Value được tính toán và lưu trữ hiệu quả qua cơ chế **Zero-Shot KV-cache Sharing**. 
* **Cơ chế hoạt động:** Model không tính lại KV của các token lịch sử ở mỗi vòng lặp dọc $i$. Thay vào đó, nó ghi đè và chỉ lưu trữ trạng thái KV của vòng lặp gần nhất (most recent recurrence) cho token hiện tại đang sinh, và tái sử dụng KV cuối cùng của các token lịch sử. 
* **Trích dẫn paper (arXiv:2502.05171):**
  > *"By enabling zero-shot KV-cache sharing, the model can store only the KV state of the most recent recurrence for each token position, rather than retaining all states from previous recurrences. This allows the model to achieve performance comparable to baselines while using significantly less memory—potentially reducing memory usage by a factor of $r$."*

### Q2: Nếu Key/Value được share cố định, attention ở các vòng lặp sau vẫn học được gì? Cơ chế nào đảm bảo model không bị degenerate?
* **Cơ chế học:** Dù $K$ và $V$ cố định (hoặc được tái sử dụng từ cache), vector **Query ($Q$) vẫn được tính toán động** ở mỗi bước lặp $i$ dựa trên trạng thái ẩn biến đổi $s_{t, i}$. Query đại diện cho "trọng tâm tìm kiếm" hoặc "câu hỏi suy luận" của mô hình tại vòng lặp $i$. Việc thay đổi $Q$ cho phép mô hình tập trung vào các thông tin khác nhau trong ngữ cảnh tại mỗi bước lặp.
* **Chống degenerate:** Cơ chế **Input Injection** (concatenation/injection của embedding Prelude $e$ vào mọi vòng lặp) và việc liên tục cập nhật Query $Q$ thông qua các block lặp đảm bảo trạng thái ẩn không bị hội tụ về một điểm tầm thường (collapse) hay lặp lại đầu ra tĩnh.

### Q3: Universal Transformers (2018) xử lý KV như thế nào? Có share KV không hay dùng cơ chế khác?
* **Trả lời:** Universal Transformers (Dehghani et al., 2018) sử dụng một Transformer block tiêu chuẩn làm hàm chuyển đổi (transition function). Trong UT gốc, $Q$, $K$, và $V$ đều được tính toán lại hoàn toàn từ trạng thái ẩn của bước lặp trước đó ở mỗi bước lặp dọc. 
* **Điểm khác biệt:** UT gốc **không** chia sẻ cố định KV qua các bước lặp dọc trong quá trình huấn luyện, dẫn đến việc tốn bộ nhớ KV-cache gấp $r$ lần lúc inference hoặc gặp lỗi training-inference mismatch nếu ép chia sẻ zero-shot.

### Q4: Với truncated BPTT (k=8 vòng cuối có gradient), việc share KV có gây vấn đề gì với gradient flow không? Giải thích cơ chế.
* **Trả lời:** Việc share KV **không** gây vấn đề nghẽn mà trái lại còn tạo ra một **lộ trình gradient cực kỳ thông thoáng (gradient highway)**.
* **Cơ chế:** Khi $K$ và $V$ được tính một lần duy nhất ở đầu (ví dụ $i=0$ hoặc từ Prelude $e$) và dùng chung cho các bước lặp sau, gradient từ $k$ vòng lặp cuối cùng khi lan truyền ngược sẽ chảy trực tiếp về duy nhất một phép chiếu $K, V$ ban đầu. Các dòng gradient này sẽ được **tích lũy (accumulated)** tại phép chiếu này và truyền thẳng về Prelude/Embedding. Điều này giúp tối ưu hóa cực tốt cho các lớp embedding và chiếu thông tin đầu vào.

### Q5: Có rủi ro kỹ thuật nào khi implement SS-KVP vào kiến trúc Bigram hiện tại không?
1. **Lệch pha huấn luyện - kiểm thử (Train-Inference Mismatch):** Nếu trong lúc train ta tính KV động từng vòng nhưng lúc sinh ta lại đóng băng KV của quá khứ, mô hình sẽ bị suy giảm chất lượng. Giải pháp là thiết lập phép chiếu $K, V$ hoàn toàn tĩnh (chỉ tính 1 lần duy nhất từ Prelude hoặc $s_0$) trong cả quá trình train và eval.
2. **Gradient tích lũy quá lớn:** Việc tích lũy gradient từ nhiều vòng lặp về một lớp chiếu duy nhất có thể gây bùng nổ gradient. Cần kiểm soát chặt chẽ bằng `grad_clip` (khuyến nghị là 1.0 hoặc 0.2).

---

## ## RESEARCH 1B: Xác minh BMSSP implementation

### Q6: SentencePiece byte-fallback hoạt động chính xác như thế nào? Khi nào dùng byte, khi nào dùng subword?
* **Cơ chế:** SentencePiece xây dựng một từ điển gồm các subwords thông thường và 256 byte UTF-8 thô (từ `<0x00>` đến `<0xFF>`).
* **Quy tắc sử dụng:**
  * Khi gặp ký tự nằm trong từ điển subwords $\rightarrow$ Mã hóa thành subword token (ưu tiên độ dài dài nhất/xác suất cao nhất).
  * Khi gặp ký tự lạ, emoji hoặc từ ngoại ngữ ngoài từ điển (OOV) $\rightarrow$ Tự động phân rã ký tự đó thành chuỗi các bytes UTF-8 thô và dùng các byte tokens tương ứng để biểu diễn. Loại bỏ hoàn toàn lỗi `<unk>`.

### Q7: Có thể ép SentencePiece phân đoạn tại ranh giới âm tiết tiếng Việt không? Cơ chế nào để làm điều đó?
* **Trả lời:** Có thể làm được một cách hoàn hảo bằng cách cấu hình **`split_by_whitespace=True`** (đây là mặc định của SentencePiece).
* **Cơ chế:** Trong tiếng Việt, các âm tiết được phân tách rõ ràng bởi dấu cách. Khi bật `split_by_whitespace=True`, SentencePiece sẽ không bao giờ thực hiện phép gộp (merge) hai ký tự nằm ở hai phía của dấu cách thành một token đơn. Như vậy, ranh giới âm tiết được bảo tồn tuyệt đối.

### Q8: Vocab size tối ưu cho tiếng Việt với byte-fallback là bao nhiêu? Llama 3 và Gemma 2 dùng bao nhiêu?
* **Vocab size tối ưu cho Bigram:** **8,000** là con số lý tưởng cho một mô hình tiếng Việt cực nhỏ. Nó bao phủ đầy đủ tất cả các âm tiết tiếng Việt thông dụng mà không làm phình bảng embedding (chỉ tốn khoảng 4MB tham số).
* **Quy mô lớn:** Llama 3 dùng 128k, Gemma 2 dùng 256k (đây là các tokenizer đa ngôn ngữ khổng lồ nên cần dung lượng lớn để chứa các hệ chữ viết khác như Trung, Nhật, Hàn, Ả Rập).

### Q9: Thư viện sentencepiece của Python có hỗ trợ đủ tính năng cần thiết không? Version nào?
* **Trả lời:** Có hỗ trợ đầy đủ. Thư viện `sentencepiece` phiên bản mới nhất (từ 0.1.99 trở lên) hỗ trợ đầy đủ tính năng huấn luyện Unigram, byte-fallback thông qua API `sentencepiece.SentencePieceTrainer.train()`.

### Q10: Có cần normalize Unicode (NFC/NFD/NFKC) trước khi train tokenizer không? Llama 3 và Gemma 2 làm gì?
* **Trả lời:** **Rất cần thiết.** Tiếng Việt có hai kiểu gõ dấu (Tổ hợp - NFD và Dựng sẵn - NFC). 
* **Quy chuẩn:** Bắt buộc normalize về **NFC** trước khi train và encode để tránh tình trạng một chữ (ví dụ "á") bị biểu diễn bằng hai bộ token khác nhau, làm phân mảnh dữ liệu. Llama 3 và Gemma 2 đều chuẩn hóa văn bản về NFC/NFKC trước khi đưa vào tokenizer.
