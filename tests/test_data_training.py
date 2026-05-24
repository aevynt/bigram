"""
test_data_training.py
=====================
Bộ kiểm thử cho pipeline dữ liệu và quá trình huấn luyện.

Chạy bằng:  pytest tests/test_data_training.py -v
Hoặc:       python tests/test_data_training.py

Bao phủ:
  - prepare_corpus tạo đúng file nhị phân.
  - PackedDataset cắt block và tạo target đúng.
  - Optimizer và LR scheduler hoạt động.
  - Một vòng training ngắn: loss thực sự giảm.
  - Lưu / nạp checkpoint.
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.config import tiny_config
from bigram.model import BigramModel
from bigram.tokenizer import BigramTokenizer
from bigram.data import prepare_corpus, PackedDataset
from bigram.training import Trainer, build_optimizer, get_lr

_SAMPLE = ("""Trí tuệ nhân tạo do người Việt phát triển sẽ phục vụ đất nước.
Hà Nội mùa thu lá vàng rơi đầy trên những con phố nhỏ.
Học tập là con đường ngắn nhất dẫn tới thành công bền vững.
""" * 40)


def _setup_data(tmp=None):
    """Tạo tokenizer + dữ liệu nhị phân cho test."""
    if tmp is None:
        tmp = tempfile.gettempdir()
    txt = os.path.join(tmp, "_test_dt_corpus.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(_SAMPLE)
    tok = BigramTokenizer.train([txt], vocab_size=256, min_frequency=1)
    prefix = os.path.join(tmp, "_test_dt_data")
    stats = prepare_corpus(txt, tok, prefix)
    return tok, prefix, stats


def test_prepare_corpus():
    """prepare_corpus phải tạo file .tok.bin và .tone.bin."""
    tok, prefix, stats = _setup_data()
    assert os.path.exists(prefix + ".tok.bin")
    assert os.path.exists(prefix + ".tone.bin")
    assert stats["n_tokens"] > 0
    print(f"  [OK] test_prepare_corpus ({stats['n_tokens']} token)")


def test_packed_dataset():
    """PackedDataset cắt block đúng và target = token dịch trái."""
    tok, prefix, _ = _setup_data()
    ds = PackedDataset(prefix + ".tok.bin", prefix + ".tone.bin", block_size=32)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["token_ids"].shape == (32,)
    assert sample["tone_ids"].shape == (32,)
    assert sample["targets"].shape == (32,)
    # Target phải bằng token dịch trái 1 vị trí.
    assert torch.equal(sample["token_ids"][1:], sample["targets"][:-1])
    print("  [OK] test_packed_dataset")


def test_optimizer_param_groups():
    """Optimizer phải tách 2 nhóm: có và không có weight decay."""
    cfg = tiny_config()
    model = BigramModel(cfg.model)
    opt = build_optimizer(model, cfg.train)
    assert len(opt.param_groups) == 2
    # Nhóm 0 có weight decay, nhóm 1 không.
    assert opt.param_groups[0]["weight_decay"] > 0
    assert opt.param_groups[1]["weight_decay"] == 0
    print("  [OK] test_optimizer_param_groups")


def test_lr_schedule():
    """LR scheduler: warmup tăng dần, sau đó giảm theo cosine."""
    cfg = tiny_config()
    cfg.train.warmup_steps = 10
    cfg.train.max_steps = 100
    lr_start = get_lr(0, cfg.train)
    lr_warmup_end = get_lr(10, cfg.train)
    lr_end = get_lr(99, cfg.train)
    # Trong warmup, LR phải tăng.
    assert lr_start < lr_warmup_end
    # Sau warmup, LR phải giảm.
    assert lr_end < lr_warmup_end
    print("  [OK] test_lr_schedule")


def test_training_loss_decreases():
    """Một vòng train ngắn: loss cuối phải thấp hơn loss đầu."""
    torch.manual_seed(0)
    tok, prefix, _ = _setup_data()
    cfg = tiny_config()
    cfg.train.max_steps = 50
    cfg.train.log_interval = 100   # tắt log cho gọn.
    cfg.train.eval_interval = 100
    cfg.train.save_interval = 100
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_dt_ckpt")

    ds = PackedDataset(prefix + ".tok.bin", prefix + ".tone.bin",
                       block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = Trainer(model, cfg, ds)

    # Đo loss ban đầu.
    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    loss_before = trainer._forward_loss(batch)["lm"].item()

    trainer.train()

    # Đo loss sau khi train.
    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    loss_after = trainer._forward_loss(batch)["lm"].item()

    assert loss_after < loss_before, \
        f"Loss không giảm: {loss_before:.3f} -> {loss_after:.3f}"
    print(f"  [OK] test_training_loss_decreases "
          f"({loss_before:.3f} -> {loss_after:.3f})")


def test_checkpoint_save_load():
    """Lưu checkpoint rồi nạp lại: trọng số phải khớp."""
    torch.manual_seed(0)
    tok, prefix, _ = _setup_data()
    cfg = tiny_config()
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_dt_ckpt2")
    ds = PackedDataset(prefix + ".tok.bin", prefix + ".tone.bin",
                       block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = Trainer(model, cfg, ds)
    path = trainer.save_checkpoint("test.pt")

    # Tạo model mới, nạp lại.
    model2 = BigramModel(cfg.model)
    trainer2 = Trainer(model2, cfg, ds)
    trainer2.load_checkpoint(path)

    # So sánh một tham số bất kỳ.
    p1 = dict(model.named_parameters())["token_embedding.weight"]
    p2 = dict(model2.named_parameters())["token_embedding.weight"]
    assert torch.equal(p1, p2)
    print("  [OK] test_checkpoint_save_load")


def run_all():
    print("Đang chạy test_data_training...")
    test_prepare_corpus()
    test_packed_dataset()
    test_optimizer_param_groups()
    test_lr_schedule()
    test_training_loss_decreases()
    test_checkpoint_save_load()
    print("test_data_training: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
