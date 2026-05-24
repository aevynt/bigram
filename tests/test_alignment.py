"""
test_alignment.py
=================
Bộ kiểm thử cho các giai đoạn alignment của pipeline:
  - SFT          (Supervised Fine-Tuning).
  - DPO          (Direct Preference Optimization).
  - Calibration  (huấn luyện abstention head).

Chạy bằng:  pytest tests/test_alignment.py -v
Hoặc:       python tests/test_alignment.py

Lưu ý quan trọng: các test này kiểm tra CƠ CHẾ chạy đúng (loss giảm, gradient
hợp lệ, mask đúng) — KHÔNG kiểm tra chất lượng model, vì điều đó cần GPU và
dữ liệu thật. Đây là điều kiện cần, không phải điều kiện đủ.
"""

import os
import sys
import json
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.config import tiny_config
from bigram.model import BigramModel
from bigram.tokenizer import BigramTokenizer
from bigram.data import JsonlSFTDataset, PreferenceDataset, CalibrationDataset
from bigram.training import Trainer, DPOTrainer
from bigram.training.dpo import sequence_logprob, dpo_loss

_CORPUS = ("""Hà Nội là thủ đô của Việt Nam ngàn năm văn hiến.
Học tập chăm chỉ là con đường dẫn tới thành công bền vững.
Trí tuệ nhân tạo giúp con người giải quyết nhiều vấn đề khó.
""" * 30)


def _get_tokenizer(tmp=None):
    if tmp is None:
        tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "_test_align_corpus.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_CORPUS)
    return BigramTokenizer.train([path], vocab_size=256, min_frequency=1)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tiny_cfg(tok):
    cfg = tiny_config()
    cfg.model.vocab_size = tok.vocab_size
    return cfg


# ----------------------------------------------------------------------
# SFT
# ----------------------------------------------------------------------
def test_sft_dataset_masks_prompt():
    """JsonlSFTDataset phải mask phần prompt khỏi target (= ignore_index)."""
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_sft.jsonl")
    _write_jsonl(path, [
        {"prompt": "Thủ đô Việt Nam là gì?", "response": "Là Hà Nội."},
    ])
    ds = JsonlSFTDataset(path, tok, block_size=64)
    sample = ds[0]
    # Phải có ít nhất một vị trí bị mask (phần prompt) và một vị trí không bị.
    n_masked = (sample["targets"] == -100).sum().item()
    n_kept = (sample["targets"] != -100).sum().item()
    assert n_masked > 0, "Prompt không được mask"
    assert n_kept > 0, "Toàn bộ bị mask — response cũng mất"
    print("  [OK] test_sft_dataset_masks_prompt")


def test_sft_training_runs():
    """SFT chạy qua Trainer: loss phải hữu hạn và giảm."""
    torch.manual_seed(0)
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_sft2.jsonl")
    _write_jsonl(path, [
        {"prompt": "Hà Nội là gì?", "response": "Hà Nội là thủ đô Việt Nam."},
        {"prompt": "Học tập thế nào?", "response": "Học tập cần chăm chỉ."},
    ] * 8)
    cfg = _tiny_cfg(tok)
    cfg.train.max_steps = 20
    cfg.train.log_interval = 100
    cfg.train.eval_interval = 100
    cfg.train.save_interval = 100
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_sft_ckpt")

    ds = JsonlSFTDataset(path, tok, block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = Trainer(model, cfg, ds)

    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    loss_before = trainer._forward_loss(batch)["lm"].item()
    trainer.train()
    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    loss_after = trainer._forward_loss(batch)["lm"].item()

    assert loss_after < loss_before, \
        f"SFT loss không giảm: {loss_before:.3f} -> {loss_after:.3f}"
    print(f"  [OK] test_sft_training_runs ({loss_before:.3f} -> {loss_after:.3f})")


# ----------------------------------------------------------------------
# DPO
# ----------------------------------------------------------------------
def test_sequence_logprob():
    """sequence_logprob trả về (batch,) và bỏ qua vị trí ignore_index."""
    torch.manual_seed(0)
    logits = torch.randn(2, 10, 50)
    targets = torch.randint(0, 50, (2, 10))
    lp_full = sequence_logprob(logits, targets)
    assert lp_full.shape == (2,)
    # Mask hết -> log-prob phải bằng 0.
    targets_masked = torch.full_like(targets, -100)
    lp_masked = sequence_logprob(logits, targets_masked)
    assert torch.allclose(lp_masked, torch.zeros(2))
    print("  [OK] test_sequence_logprob")


def test_dpo_loss_basic():
    """DPO loss: khi chosen tốt hơn rõ rệt, loss phải nhỏ và acc cao."""
    # chosen có log-prob cao hơn ref, rejected thấp hơn ref -> model "đúng".
    pol_chosen = torch.tensor([2.0, 2.0])
    pol_rejected = torch.tensor([-2.0, -2.0])
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    out = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert out["loss"].item() > 0
    assert out["reward_accuracy"].item() == 1.0
    assert out["reward_margin"].item() > 0
    print(f"  [OK] test_dpo_loss_basic (loss={out['loss'].item():.4f})")


def test_dpo_loss_gradient():
    """DPO loss phải lan gradient về model policy."""
    torch.manual_seed(0)
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_dpo.jsonl")
    _write_jsonl(path, [
        {"prompt": "Hà Nội là gì?",
         "chosen": "Hà Nội là thủ đô của Việt Nam.",
         "rejected": "Hà Nội là một loại trái cây."},
    ])
    cfg = _tiny_cfg(tok)
    ds = PreferenceDataset(path, tok, block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)

    from bigram.training.dpo import compute_dpo_loss_from_batch
    import copy
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=1)))
    out = compute_dpo_loss_from_batch(model, ref, batch, beta=0.1)
    out["loss"].backward()

    # Model policy phải nhận gradient.
    g = model.token_embedding.weight.grad
    assert g is not None and torch.isfinite(g).all()
    # Model reference KHÔNG được nhận gradient (nó đông cứng).
    assert ref.token_embedding.weight.grad is None
    print("  [OK] test_dpo_loss_gradient")


def test_dpo_trainer_runs():
    """DPOTrainer chạy trọn vẹn vài bước, loss hữu hạn."""
    torch.manual_seed(0)
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_dpo2.jsonl")
    _write_jsonl(path, [
        {"prompt": "Hà Nội là gì?",
         "chosen": "Hà Nội là thủ đô Việt Nam.",
         "rejected": "Tôi không biết gì cả."},
        {"prompt": "Học tập ra sao?",
         "chosen": "Học tập cần sự chăm chỉ và kiên trì.",
         "rejected": "Học tập không quan trọng."},
    ] * 6)
    cfg = _tiny_cfg(tok)
    cfg.train.max_steps = 15
    cfg.train.log_interval = 100
    cfg.train.save_interval = 100
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_dpo_ckpt")

    ds = PreferenceDataset(path, tok, block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = DPOTrainer(model, cfg, ds, beta=0.1)
    final = trainer.train()
    assert os.path.exists(final)
    print("  [OK] test_dpo_trainer_runs")


def test_dpo_ref_model_frozen():
    """Model tham chiếu trong DPOTrainer phải bị đông cứng hoàn toàn."""
    torch.manual_seed(0)
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_dpo3.jsonl")
    _write_jsonl(path, [
        {"prompt": "A?", "chosen": "Câu trả lời tốt.",
         "rejected": "Câu trả lời tệ."},
    ] * 4)
    cfg = _tiny_cfg(tok)
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_dpo_ckpt2")
    ds = PreferenceDataset(path, tok, block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = DPOTrainer(model, cfg, ds, beta=0.1)
    # Mọi tham số của ref đều phải requires_grad=False.
    assert all(not p.requires_grad for p in trainer.ref.parameters())
    print("  [OK] test_dpo_ref_model_frozen")


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
def test_calibration_dataset():
    """CalibrationDataset trả về nhãn abstention đúng định dạng."""
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_cal.jsonl")
    _write_jsonl(path, [
        {"prompt": "Câu hỏi thường?", "response": "Câu trả lời.",
         "should_abstain": 0},
        {"prompt": "Câu hỏi bịa?", "response": "Tôi không chắc.",
         "should_abstain": 1},
    ])
    ds = CalibrationDataset(path, tok, block_size=64)
    s0 = ds[0]  # should_abstain = 0.
    s1 = ds[1]  # should_abstain = 1.
    assert "abstention_targets" in s0 and "abstention_mask" in s0
    # Mẫu 0: mọi nhãn abstention ở vùng mask phải = 0.
    masked0 = s0["abstention_targets"][s0["abstention_mask"] > 0]
    assert (masked0 == 0).all()
    # Mẫu 1: mọi nhãn abstention ở vùng mask phải = 1.
    masked1 = s1["abstention_targets"][s1["abstention_mask"] > 0]
    assert (masked1 == 1).all()
    print("  [OK] test_calibration_dataset")


def test_calibration_training_runs():
    """Calibration qua Trainer: abstention head phải nhận gradient và học."""
    torch.manual_seed(0)
    tok = _get_tokenizer()
    path = os.path.join(tempfile.gettempdir(), "_test_cal2.jsonl")
    _write_jsonl(path, [
        {"prompt": "Thủ đô Việt Nam?", "response": "Hà Nội.",
         "should_abstain": 0},
        {"prompt": "Số nhà tôi là?", "response": "Tôi không biết.",
         "should_abstain": 1},
    ] * 8)
    cfg = _tiny_cfg(tok)
    cfg.model.abstention_loss_coef = 1.0
    cfg.train.max_steps = 20
    cfg.train.log_interval = 100
    cfg.train.eval_interval = 100
    cfg.train.save_interval = 100
    cfg.train.out_dir = os.path.join(tempfile.gettempdir(), "_test_cal_ckpt")

    ds = CalibrationDataset(path, tok, block_size=cfg.model.max_seq_len)
    model = BigramModel(cfg.model)
    trainer = Trainer(model, cfg, ds)

    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    abst_before = trainer._forward_loss(batch)["abstention"].item()
    trainer.train()
    batch = trainer._move_batch(next(iter(trainer.train_loader)))
    abst_after = trainer._forward_loss(batch)["abstention"].item()

    # Abstention loss phải > 0 (head đang được train) và nên giảm.
    assert abst_before > 0
    assert abst_after < abst_before, \
        f"Abstention loss không giảm: {abst_before:.3f} -> {abst_after:.3f}"
    print(f"  [OK] test_calibration_training_runs "
          f"({abst_before:.3f} -> {abst_after:.3f})")


def run_all():
    print("Đang chạy test_alignment...")
    test_sft_dataset_masks_prompt()
    test_sft_training_runs()
    test_sequence_logprob()
    test_dpo_loss_basic()
    test_dpo_loss_gradient()
    test_dpo_trainer_runs()
    test_dpo_ref_model_frozen()
    test_calibration_dataset()
    test_calibration_training_runs()
    print("test_alignment: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
