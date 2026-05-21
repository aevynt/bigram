# Triết lý thiết kế Bigram

Bigram là một mô hình ngôn ngữ tiếng Việt được thiết kế để **huấn luyện từ đầu** (from scratch), không phải fine-tune một model có sẵn. Tài liệu này giải thích *vì sao* kiến trúc được dựng như hiện tại.

## Bối cảnh

Mục tiêu của Bigram không phải là chạy đua tham số với các model lớn, mà là một kiến trúc *đúng cho usecase tiếng Việt*: hội thoại tự nhiên, suy luận logic, xử lý văn bản dài, và trên hết — **nói không với hallucination**. Người dùng mục tiêu là tác tử AI, chuyên viên nghiên cứu, tổ chức nhà nước: những nơi một câu trả lời sai tự tin còn tệ hơn một câu "tôi không chắc".

Triết lý cốt lõi: *ưu tiên kiến trúc tốt, rồi mới scale compute dần*. Một model nhỏ nhưng "biết nghĩ" và "biết im lặng" có giá trị thực hơn một model lớn hay bịa.

## Ba trụ cột kiến trúc

### Trụ cột 1 — Recurrent depth: suy luận sâu mà ít tham số

Transformer thường có độ sâu cố định bằng số layer. Muốn "nghĩ sâu hơn" phải thêm layer, tức thêm tham số, tức thêm dữ liệu và compute.

Bigram tách model thành ba nhóm: **Prelude** (đưa input vào không gian latent), **Recurrent core** (khối lõi), và **Coda** (giải mã ra dự đoán). Điểm mấu chốt: khối Recurrent được **lặp lại** `r` vòng. Với 8 layer thật và `r = 32`, độ sâu hiệu dụng là `2 + 4×32 + 2 = 132` layer — sâu hơn cả những transformer cố định lớn nhất, nhưng vẫn chỉ 8 layer tham số.

Hệ quả thực dụng: `r` có thể tăng lúc suy luận. Câu hỏi khó thì cho model "nghĩ" nhiều vòng hơn. Đây là một trục scale mới — scale *test-time compute* — bên cạnh scale tham số và scale dữ liệu.

Hai chi tiết kỹ thuật giữ cho recurrence hoạt động:

- **Input injection**: embedding từ Prelude được đưa vào *mọi* vòng lặp, không chỉ vòng đầu. Nếu chỉ đưa vào đầu, quá trình lặp sẽ chỉ phụ thuộc điều kiện biên và không ổn định. Việc tái đưa input giống như thuật toán tối ưu lặp phải nhìn lại dữ liệu ở mỗi bước.
- **Truncated backpropagation**: lúc train, forward chạy đủ `r` vòng nhưng gradient chỉ truyền qua `k = 8` vòng cuối. Nhờ vậy bộ nhớ huấn luyện *độc lập* với `r` — có thể train với `r` lấy từ phân phối đuôi dài mà chi phí không đổi.

### Trụ cột 2 — Abstention head: biết khi nào nên im lặng

Hầu hết hallucination không phải lỗi kiến trúc mà là lỗi *động lực*: model được thưởng khi đoán, không bị phạt khi đoán sai, nên nó học cách luôn đoán.

Bigram thêm một đầu ra thứ hai — **abstention head** — song song với đầu ra dự đoán token. Đầu này dự đoán xác suất model *nên từ chối* trả lời. Lúc suy luận, nếu xác suất này vượt ngưỡng, model dừng lại thay vì bịa.

Trụ cột này chỉ hoàn chỉnh khi kết hợp với **hàm thưởng thành thật** ở giai đoạn calibration: trả lời đúng được +1, nói "không chắc" được 0 (không phạt), trả lời sai tự tin bị phạt nặng. Cấu trúc thưởng này phá vỡ động lực đoán bừa ngay từ gốc.

### Trụ cột 3 — Tokenizer nhận biết thanh điệu

Tiếng Việt có 6 thanh điệu. Một tokenizer thông thường coi "ma / má / mà / mả / mã / mạ" gần như sáu token rời rạc, dù chúng chia sẻ cùng một âm gốc. Điều này làm phình từ điển và che mất quy luật âm vị học.

Bigram tách mỗi âm tiết thành hai luồng: **âm gốc** (đã bỏ dấu thanh) và **thanh điệu**. Model có hai bảng embedding riêng, cộng lại khi nhúng. Như vậy "má" và "mà" chia sẻ toàn bộ tri thức về âm gốc "ma", chỉ khác ở một embedding thanh điệu nhỏ.

Lưu ý quan trọng: chỉ tách *dấu thanh* (sắc, huyền, hỏi, ngã, nặng), không tách *dấu tạo nguyên âm* (ă, â, ê, ô, ơ, ư, đ) — vì ă/â/ê là âm vị riêng biệt, còn dấu thanh mới là thứ lặp lại có quy luật.

Tính đối xứng được giữ ở cả đầu ra: vì đầu vào có hai luồng, đầu ra cũng có hai luồng. Bên cạnh `lm_head` dự đoán âm gốc, model có thêm một **tone head** dự đoán thanh điệu của token tiếp theo. Nhờ vậy văn bản sinh ra giữ được dấu thanh đầy đủ — nếu chỉ dự đoán âm gốc, kết quả sẽ là tiếng Việt không dấu, vừa khó đọc vừa mơ hồ về nghĩa.

## Các lựa chọn hỗ trợ

Ngoài ba trụ cột, Bigram dùng một số kỹ thuật đã được kiểm chứng để giữ ổn định ở quy mô lớn: **Grouped Query Attention** (giảm bộ nhớ KV-cache cho văn bản dài), **Mixture of Experts** với một "Vietnamese expert" luôn-bật (tăng dung lượng mà không tăng compute mỗi token), **sandwich normalization** và **LayerScale** (giữ recurrence không bị latent collapse), và **RoPE** (tổng quát hóa ra chuỗi dài).

## Pipeline 5 giai đoạn

Bigram được huấn luyện qua năm giai đoạn: (0) train tokenizer, (1) pre-training với recurrent depth, (2) mid-training tập trung suy luận, (3) alignment bằng SFT và DPO, (4) calibration để dạy abstention. Codebase này hiện thực hóa đầy đủ giai đoạn 0–1, và cung cấp sẵn `JsonlSFTDataset` cùng abstention head cho giai đoạn 3–4.
