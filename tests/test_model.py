"""
test_model.py
=============
Bộ kiểm thử cho kiến trúc model Bigram.

Chạy bằng:  pytest tests/test_model.py -v
Hoặc chạy trực tiếp: python tests/test_model.py

Các test bao phủ:
  - Forward pass cho ra đúng shape.
  - Backward pass: gradient lan tới mọi tham số, không có NaN.
  - Truncated BPTT: train được với r lớn hơn k.
  - Recurrence thực sự thay đổi output.
  - Abstention head được train khi có nhãn.
  - Sinh văn bản (generate) chạy được.
"""

import os
import sys

import torch

# Cho phép import package `bigram`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.config import tiny_config
from bigram.model import BigramModel, compute_total_loss, sample_recurrence


def _make_model():
    """Tạo một model nhỏ + dữ liệu giả để test."""
    torch.manual_seed(0)
    cfg = tiny_config()
    model = BigramModel(cfg.model)
    return model, cfg


def test_forward_shape():
    """Forward pass phải trả về logits, abstention và tone đúng kích thước."""
    model, cfg = _make_model()
    model.eval()
    b, s = 2, 16
    tok = torch.randint(0, cfg.model.vocab_size, (b, s))
    tone = torch.randint(0, cfg.model.tone_vocab_size, (b, s))

    out = model(tok, tone)
    assert out["logits"].shape == (b, s, cfg.model.vocab_size)
    assert out["abstention_logits"].shape == (b, s)
    assert out["tone_logits"].shape == (b, s, cfg.model.tone_vocab_size)
    assert out["num_recurrence"] >= 1
    print("  [OK] test_forward_shape")


def test_forward_without_tone():
    """Model phải chạy được kể cả khi không có thông tin thanh điệu."""
    model, cfg = _make_model()
    model.eval()
    tok = torch.randint(0, cfg.model.vocab_size, (2, 16))
    out = model(tok, tone_ids=None)
    assert out["logits"].shape == (2, 16, cfg.model.vocab_size)
    print("  [OK] test_forward_without_tone")


def test_backward_gradient_flow():
    """Backward: mọi tham số (trừ abstention head) phải nhận gradient."""
    model, cfg = _make_model()
    model.train()
    b, s = 2, 16
    tok = torch.randint(0, cfg.model.vocab_size, (b, s))
    tone = torch.randint(0, cfg.model.tone_vocab_size, (b, s))
    targets = torch.randint(0, cfg.model.vocab_size, (b, s))
    # Cung cấp cả tone_targets để tone head cũng được train và kiểm tra.
    tone_targets = torch.randint(0, cfg.model.tone_vocab_size, (b, s))

    out = model(tok, tone)
    losses = compute_total_loss(out, targets, cfg.model,
                                tone_targets=tone_targets)
    losses["total"].backward()

    # Abstention head KHÔNG có gradient ở đây là đúng — nó chỉ được train
    # ở giai đoạn calibration (khi có nhãn abstention).
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "abstention_head" in name:
            continue
        assert p.grad is not None, f"Tham số {name} không có gradient"
        assert not torch.isnan(p.grad).any(), f"Gradient NaN ở {name}"
    print("  [OK] test_backward_gradient_flow")


def test_abstention_head_trains():
    """Khi có nhãn abstention, abstention head phải nhận gradient."""
    model, cfg = _make_model()
    model.train()
    b, s = 2, 16
    tok = torch.randint(0, cfg.model.vocab_size, (b, s))
    targets = torch.randint(0, cfg.model.vocab_size, (b, s))
    abst_tgt = torch.randint(0, 2, (b, s)).float()
    abst_mask = torch.ones(b, s)

    out = model(tok)
    losses = compute_total_loss(out, targets, cfg.model,
                                abstention_targets=abst_tgt,
                                abstention_mask=abst_mask)
    losses["total"].backward()
    assert model.abstention_head.weight.grad is not None
    assert losses["abstention"] > 0
    print("  [OK] test_abstention_head_trains")


def test_tone_head_trains():
    """Khi có nhãn thanh điệu, tone head phải nhận gradient và tone loss > 0."""
    model, cfg = _make_model()
    model.train()
    b, s = 2, 16
    tok = torch.randint(0, cfg.model.vocab_size, (b, s))
    targets = torch.randint(0, cfg.model.vocab_size, (b, s))
    tone_tgt = torch.randint(0, cfg.model.tone_vocab_size, (b, s))

    out = model(tok)
    losses = compute_total_loss(out, targets, cfg.model,
                                tone_targets=tone_tgt)
    losses["total"].backward()
    assert model.tone_head.weight.grad is not None
    assert losses["tone"] > 0
    print("  [OK] test_tone_head_trains")


def test_recurrence_changes_output():
    """Số vòng lặp khác nhau phải cho output khác nhau."""
    model, cfg = _make_model()
    model.eval()
    tok = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        o1 = model(tok, num_recurrence=1)["logits"]
        o8 = model(tok, num_recurrence=8)["logits"]
    diff = (o1 - o8).abs().mean().item()
    assert diff > 1e-4, "Recurrence không thay đổi output — khối lõi vô tác dụng"
    print(f"  [OK] test_recurrence_changes_output (diff={diff:.4f})")


def test_eval_forward_is_deterministic():
    """Eval forward should be deterministic with the default zero latent start."""
    model, cfg = _make_model()
    model.eval()
    tok = torch.randint(0, cfg.model.vocab_size, (2, 16))
    tone = torch.randint(0, cfg.model.tone_vocab_size, (2, 16))
    with torch.no_grad():
        o1 = model(tok, tone, num_recurrence=4)["logits"]
        o2 = model(tok, tone, num_recurrence=4)["logits"]
    assert torch.allclose(o1, o2), "Eval forward changed despite identical inputs"
    print("  [OK] test_eval_forward_is_deterministic")


def test_truncated_bptt_large_r():
    """Truncated BPTT: train được với r lớn hơn nhiều so với k."""
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.model.backprop_depth = 3  # k = 3
    model = BigramModel(cfg.model)
    model.train()
    tok = torch.randint(0, cfg.model.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.model.vocab_size, (2, 16))

    # r = 20 >> k = 3: 17 vòng đầu chạy no_grad.
    out = model(tok, num_recurrence=20)
    loss = compute_total_loss(out, targets, cfg.model)["total"]
    loss.backward()

    g = model.recurrent_adapter.weight.grad
    assert g is not None and torch.isfinite(g).all()
    print("  [OK] test_truncated_bptt_large_r")


def test_sample_recurrence_distribution():
    """Hàm sample_recurrence phải luôn trả về số nguyên >= 1."""
    for _ in range(100):
        r = sample_recurrence(mean_r=32.0, sigma=0.5)
        assert isinstance(r, int) and r >= 1
    print("  [OK] test_sample_recurrence_distribution")


def test_generate_runs():
    """Hàm generate phải sinh đúng số token và kèm dãy thanh điệu."""
    model, cfg = _make_model()
    model.eval()
    tok = torch.randint(0, cfg.model.vocab_size, (1, 5))
    out, out_tones, abstained = model.generate(
        tok, max_new_tokens=10, num_recurrence=4)
    # Độ dài mới = độ dài cũ + số token sinh (trừ khi abstain sớm).
    assert out.shape[1] <= 5 + 10
    assert out.shape[1] >= 5
    # Khi model có tone head, generate phải trả về dãy thanh điệu cùng độ dài
    # với dãy token (mỗi token có một thanh điệu tương ứng).
    assert out_tones is not None
    assert out_tones.shape == out.shape
    # Thanh điệu phải là id hợp lệ (trong khoảng từ điển thanh điệu).
    assert out_tones.min() >= 0
    assert out_tones.max() < cfg.model.tone_vocab_size
    print("  [OK] test_generate_runs")


def test_generate_with_nucleus_and_repetition_penalty():
    """generate() should support top-p and repetition penalty sampling."""
    model, cfg = _make_model()
    model.eval()
    torch.manual_seed(0)
    tok = torch.randint(0, cfg.model.vocab_size, (1, 5))
    out, out_tones, _ = model.generate(
        tok,
        max_new_tokens=5,
        num_recurrence=4,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    assert out.shape[1] >= 5
    assert out.shape == out_tones.shape
    print("  [OK] test_generate_with_nucleus_and_repetition_penalty")


def test_param_count_reasonable():
    """Model tiny phải có số tham số hợp lý (không rỗng, không khổng lồ)."""
    model, cfg = _make_model()
    n = model.num_parameters()
    assert 10_000 < n < 10_000_000
    print(f"  [OK] test_param_count_reasonable ({n} tham số)")


def run_all():
    """Chạy toàn bộ test theo thứ tự (dùng khi không có pytest)."""
    print("Đang chạy test_model...")
    test_forward_shape()
    test_forward_without_tone()
    test_backward_gradient_flow()
    test_abstention_head_trains()
    test_tone_head_trains()
    test_recurrence_changes_output()
    test_eval_forward_is_deterministic()
    test_truncated_bptt_large_r()
    test_sample_recurrence_distribution()
    test_generate_runs()
    test_generate_with_nucleus_and_repetition_penalty()
    test_param_count_reasonable()
    print("test_model: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
