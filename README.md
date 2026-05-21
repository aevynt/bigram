# Bigram

Mô hình ngôn ngữ tiếng Việt với kiến trúc **recurrent-depth**, thiết kế để huấn luyện từ đầu — không fine-tune model có sẵn.

Ba đặc trưng cốt lõi: khối transformer được **lặp lại** để suy luận sâu mà ít tham số; một **abstention head** dạy model biết khi nào nên nói "tôi không chắc" thay vì bịa; và một **tokenizer tách thanh điệu** riêng cho tiếng Việt. Chi tiết triết lý nằm trong `PHILOSOPHY.md`.

## Cài đặt

```bash
pip install -r requirements.txt
```

Cần Python 3.9 trở lên. PyTorch là thư viện nặng nhất; nếu có GPU NVIDIA, hãy cài bản CUDA tương ứng theo hướng dẫn ở pytorch.org.

## Kiểm tra codebase

Trước khi train, chạy bộ test để chắc chắn mọi thứ hoạt động:

```bash
python tests/run_all.py
```

Toàn bộ test chạy trên CPU trong khoảng một phút. Nếu thấy "TẤT CẢ BỘ TEST ĐỀU PASS" thì codebase sẵn sàng.

## Pipeline sử dụng

Quy trình gồm bốn bước. Giả sử bạn đã có một file văn bản tiếng Việt `data/corpus.txt` (mỗi dòng một câu hoặc đoạn).

### Bước 0 — Train tokenizer

```bash
python scripts/train_tokenizer.py \
    --input data/corpus.txt \
    --output data/tokenizer.json \
    --vocab-size 32000
```

Tokenizer chỉ train một lần và cố định. Mọi bước sau phụ thuộc vào nó.

### Bước 1 — Chuẩn bị dữ liệu

Mã hóa văn bản thành file nhị phân. Chạy hai lần, một cho tập train, một cho tập validation:

```bash
python scripts/prepare_data.py \
    --tokenizer data/tokenizer.json \
    --input data/corpus.txt \
    --output-prefix data/train

python scripts/prepare_data.py \
    --tokenizer data/tokenizer.json \
    --input data/val_corpus.txt \
    --output-prefix data/val
```

### Bước 2 — Huấn luyện model

```bash
python scripts/train.py \
    --train-data data/train \
    --val-data data/val \
    --tokenizer data/tokenizer.json \
    --preset small \
    --out-dir checkpoints
```

Có hai preset dựng sẵn: `tiny` (model siêu nhỏ, chạy được trên CPU, dùng để thử nghiệm) và `small` (khoảng 0.5B tham số, cần GPU). Để dùng cấu hình tùy chỉnh, sửa file trong `configs/` rồi truyền `--config configs/your_config.json`.

Train tiếp từ một checkpoint:

```bash
python scripts/train.py --resume checkpoints/ckpt_step1000.pt ... 
```

### Bước 3 — Sinh văn bản

```bash
python scripts/generate.py \
    --checkpoint checkpoints/ckpt_final.pt \
    --tokenizer data/tokenizer.json \
    --prompt "Thủ đô của Việt Nam là" \
    --recurrence 32 \
    --max-new-tokens 50
```

Tham số `--recurrence` điều khiển số vòng "suy nghĩ" — câu hỏi khó thì đặt cao hơn. Thêm `--abstention-threshold 0.5` để bật cơ chế chống hallucination: model sẽ dừng nếu không đủ tự tin.

## Pipeline đầy đủ — alignment

Ba bước trên cho ra một *base model* — nó tiếp nối văn bản nhưng chưa phải trợ lý hội thoại. Để hoàn thiện, có ba giai đoạn alignment, chạy lần lượt. Mỗi giai đoạn nạp checkpoint của giai đoạn trước.

### Giai đoạn 3a — SFT (Supervised Fine-Tuning)

Dạy model trả lời, từ các cặp câu hỏi/trả lời. File `.jsonl`, mỗi dòng `{"prompt": "...", "response": "..."}`.

```bash
python scripts/train_sft.py \
    --data data/sft.jsonl \
    --tokenizer data/tokenizer.json \
    --init checkpoints/ckpt_final.pt \
    --out-dir checkpoints_sft
```

### Giai đoạn 3b — DPO (Direct Preference Optimization)

Tinh chỉnh theo sở thích con người. File `.jsonl`, mỗi dòng `{"prompt": "...", "chosen": "...", "rejected": "..."}`. DPO được chọn thay cho RL/PPO vì ổn định hơn nhiều — quan trọng cho mục tiêu "AI đáng tin" của Bigram. Phải chạy *sau* SFT.

```bash
python scripts/train_dpo.py \
    --data data/preferences.jsonl \
    --tokenizer data/tokenizer.json \
    --init checkpoints_sft/ckpt_final.pt \
    --out-dir checkpoints_dpo --beta 0.1
```

### Giai đoạn 4 — Calibration

Huấn luyện abstention head — dạy model biết khi nào nên nói "tôi không chắc". Đây là bước hiện thực hóa chống hallucination, và đi cuối cùng. Tạo dữ liệu mẫu bằng `python scripts/make_calibration_data.py`.

```bash
python scripts/train_calibration.py \
    --data data/calibration.jsonl \
    --tokenizer data/tokenizer.json \
    --init checkpoints_dpo/dpo_final.pt \
    --out-dir checkpoints_final
```

Sau giai đoạn 4, dùng `generate.py` với `--abstention-threshold` để model biết từ chối khi không chắc.

## Cấu trúc thư mục

```
bigram/
├── bigram/              # Package chính
│   ├── config.py        # Toàn bộ siêu tham số
│   ├── model/           # Kiến trúc mạng nơ-ron
│   ├── tokenizer/       # Xử lý tiếng Việt (tách thanh + BPE)
│   ├── data/            # Chuẩn bị và nạp dữ liệu
│   ├── training/        # Vòng lặp huấn luyện
│   └── utils/           # Tiện ích phụ trợ
├── scripts/             # Script dòng lệnh (train, generate, ...)
├── tests/               # Bộ kiểm thử
├── configs/             # File cấu hình mẫu
├── data/                # Nơi đặt dữ liệu của bạn
├── PHILOSOPHY.md        # Triết lý thiết kế và 3 trụ cột
└── README.md            # File này
```

## Dùng như một thư viện Python

Ngoài các script, có thể import trực tiếp:

```python
from bigram import BigramModel, BigramConfig, BigramTokenizer, Trainer
from bigram import tiny_config, small_config

cfg = small_config()
model = BigramModel(cfg.model)
```

## Về dữ liệu

Codebase này **không kèm dữ liệu** — bạn cần tự chuẩn bị corpus tiếng Việt. Định dạng đơn giản: một file `.txt`, mỗi dòng một câu hoặc đoạn văn. Chất lượng và quy mô dữ liệu quyết định phần lớn chất lượng model. Để pre-training thực sự, cần ít nhất vài GB văn bản sạch.

Cho giai đoạn SFT, lớp `JsonlSFTDataset` đọc file `.jsonl` với mỗi dòng dạng `{"prompt": "...", "response": "..."}`.

## Trạng thái

Codebase hiện thực hóa **đầy đủ pipeline 5 giai đoạn**: train tokenizer, pre-training với recurrent depth, SFT, DPO, và calibration cho abstention head. Mỗi giai đoạn có script CLI riêng và đã được kiểm thử về mặt cơ chế (loss giảm, gradient hợp lệ, checkpoint nối đúng).

Một lưu ý thẳng thắn: bộ test xác nhận *cơ chế* training chạy đúng — đó là điều kiện cần, không phải điều kiện đủ. Để model thực sự "nói chuyện được" cần GPU và dữ liệu quy mô lớn, là phần việc nằm ngoài codebase này. Xem `PHILOSOPHY.md` để hiểu lý do thiết kế từng giai đoạn.
