#!/usr/bin/env python
"""
train_calibration.py
====================
Giai đoạn 4 — giai đoạn cuối của pipeline: Calibration.

Đây là bước hiện thực hóa lời hứa "nói không với hallucination". Ta huấn
luyện ABSTENTION HEAD — đầu ra dự đoán xem model có NÊN từ chối trả lời hay
không (xem PHILOSOPHY.md, trụ cột 2).

Vì sao calibration đứng SAU alignment: pre-training tự nhiên khuyến khích
model "hiệu chỉnh" (calibrated), nhưng SFT/DPO có thể làm hỏng tính chất đó.
Calibration là bước sửa lại, nên phải đi cuối cùng.

Trong giai đoạn này, loss tổng gồm hai phần: LM loss (giữ cho model không
quên cách nói) và abstention loss (dạy abstention head). Phần lớn trọng số
học tập dồn vào abstention head.

Định dạng dữ liệu — file .jsonl, mỗi dòng:
    {"prompt": "...", "response": "...", "should_abstain": 0 hoặc 1}
(tạo file mẫu bằng: python scripts/make_calibration_data.py)

Cách dùng:
    python scripts/train_calibration.py \
        --data data/calibration.jsonl \
        --tokenizer data/tokenizer.json \
        --init checkpoints_dpo/dpo_final.pt \
        --out-dir checkpoints_final
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer, BigramConfig, Trainer
from bigram.config import ModelConfig
from bigram.data import CalibrationDataset
from bigram.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Calibration cho Bigram")
    parser.add_argument("--data", required=True,
                        help="File .jsonl với prompt/response/should_abstain")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--init", required=True,
                        help="Checkpoint model đã qua SFT/DPO")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", default="checkpoints_final")
    parser.add_argument("--abstention-coef", type=float, default=1.0,
                        help="Hệ số abstention loss. Cao -> dồn lực vào head này")
    args = parser.parse_args()

    if not os.path.exists(args.init):
        print(f"LỖI: không tìm thấy checkpoint {args.init}")
        sys.exit(1)

    tokenizer = BigramTokenizer.load(args.tokenizer)

    print(f"Nạp model từ {args.init}...")
    ckpt = torch.load(args.init, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    # Bật abstention head nếu chưa bật.
    model_cfg.use_abstention_head = True
    # Tăng hệ số abstention loss — giai đoạn này tập trung vào head đó.
    model_cfg.abstention_loss_coef = args.abstention_coef

    config = BigramConfig()
    config.model = model_cfg
    model = BigramModel(model_cfg)
    # strict=False: phòng khi checkpoint cũ chưa có abstention head.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  (khởi tạo mới {len(missing)} tham số — vd abstention head)")

    config.train.max_steps = args.max_steps
    config.train.out_dir = args.out_dir
    config.train.learning_rate = 1e-5
    config.data.stage = "calibration"

    set_seed(config.train.seed)

    dataset = CalibrationDataset(args.data, tokenizer,
                                 block_size=config.model.max_seq_len)
    print(f"Số mẫu calibration: {len(dataset)}")

    # Trainer thường xử lý được — nó tự đọc abstention_targets/abstention_mask
    # từ batch (xem Trainer._forward_loss).
    trainer = Trainer(model, config, dataset)
    trainer.train()

    print()
    print("Hoàn tất pipeline! Model cuối cùng đã có abstention head được")
    print("hiệu chỉnh. Dùng scripts/generate.py với --abstention-threshold")
    print("để model biết từ chối khi không chắc chắn.")


if __name__ == "__main__":
    main()
