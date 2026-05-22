"""Tests for Bigram Tensor 1 config/model wiring."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.config import ModelConfig, tensor1_config, tiny_config
from bigram.model import BigramModel, compute_total_loss


def test_tensor1_config_exists():
    cfg = tensor1_config()
    assert cfg.model.hidden_size == 2048
    assert cfg.model.moe_scope == "recurrent_only"
    assert cfg.model.use_tool_head
    assert cfg.model.use_verifier_head
    assert cfg.train.batch_size == 1
    print("  [OK] test_tensor1_config_exists")


def test_moe_scope_validates():
    ModelConfig(moe_scope="none", use_moe=True)
    try:
        ModelConfig(moe_scope="sus")
    except AssertionError:
        print("  [OK] test_moe_scope_validates")
        return
    raise AssertionError("moe_scope invalid không bị reject")


def test_tiny_tool_verifier_forward_and_loss():
    cfg = tiny_config()
    cfg.model.use_tool_head = True
    cfg.model.use_verifier_head = True
    cfg.model.moe_scope = "recurrent_only"
    cfg.model.__post_init__()
    model = BigramModel(cfg.model)
    tok = torch.randint(0, cfg.model.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.model.vocab_size, (2, 8))
    out = model(tok, num_recurrence=2)
    assert out["tool_router_logits"].shape == (2, 8, 3)
    assert out["tool_name_logits"].shape == (2, 8, cfg.model.n_tools)
    assert out["tool_arg_hidden"].shape == (2, 8, cfg.model.hidden_size)
    assert out["verifier_logits"].shape == (2, 8)
    router = torch.full((2, 8), -100, dtype=torch.long)
    names = torch.full((2, 8), -100, dtype=torch.long)
    verifier = torch.full((2, 8), -100.0)
    router[:, 0] = 1
    names[:, 0] = 2
    verifier[:, 1] = 1.0
    loss = compute_total_loss(
        out,
        targets,
        cfg.model,
        tool_router_targets=router,
        tool_name_targets=names,
        verifier_targets=verifier,
    )
    loss["total"].backward()
    assert model.tool_head.router.weight.grad is not None
    assert model.verifier_head.weight.grad is not None
    print("  [OK] test_tiny_tool_verifier_forward_and_loss")


def run_all():
    print("Đang chạy test_tensor1_config...")
    test_tensor1_config_exists()
    test_moe_scope_validates()
    test_tiny_tool_verifier_forward_and_loss()
    print("test_tensor1_config: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
