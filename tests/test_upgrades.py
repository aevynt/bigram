"""
test_upgrades.py
================
Unit tests cho các cải tiến đột phá của Bigram V2:
  - VietnameseSyllableAwareTokenizer (VS-BPE)
  - Multi-Head Latent Attention (MLA)
  - PyTorchMambaBlock (Mamba-2 Hybrid)
  - PonderNet Dynamic Halting Loop & Halting Loss
"""

import os
import sys
import tempfile
import torch
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.config import tiny_config, bigram_v2_1_8b_a6000_config
from bigram.tokenizer import VietnameseSyllableAwareTokenizer
from bigram.model.bigram import BigramModel
from bigram.model.attention import MultiHeadLatentAttention
from bigram.model.block import PyTorchMambaBlock
from bigram.model.loss import compute_total_loss

class TestBigramV2Upgrades(unittest.TestCase):

    def test_vs_bpe_tokenizer_basic(self):
        """Xác minh bộ tách từ âm tiết VS-BPE hoạt động chuẩn xác."""
        corpus = [
            "Tiếng Việt là ngôn ngữ giàu thanh điệu.",
            "Tôi yêu những ngày hè nắng ấm.",
            "Suy luận đa bước sâu sắc ít sai."
        ]
        
        # Ghi ra file tạm để train
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
            for line in corpus:
                f.write(line + "\n")
            temp_path = f.name
            
        try:
            # Huấn luyện thử nghiệm
            tokenizer = VietnameseSyllableAwareTokenizer.train([temp_path], vocab_size=100, min_frequency=1)
            self.assertTrue(tokenizer.vocab_size > 10)
            
            # Thử encode và decode
            text = "Tôi yêu tiếng Việt."
            ids, tone = tokenizer.encode(text)
            self.assertIsNone(tone) # Single-stream
            
            decoded = tokenizer.decode(ids)
            self.assertEqual(decoded.strip(), text.strip())
            
            # Kiểm thử save/load
            save_path = temp_path + ".json"
            tokenizer.save(save_path)
            
            loaded = VietnameseSyllableAwareTokenizer.load(save_path)
            self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
            
            if os.path.exists(save_path):
                os.remove(save_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_mla_attention_shapes(self):
        """Xác minh Multi-Head Latent Attention có output shape chuẩn xác."""
        cfg = tiny_config()
        cfg.model.use_mla = True
        cfg.model.kv_latent_dim = 16
        cfg.model.decoupled_rope_dim = 8
        
        mla = MultiHeadLatentAttention(cfg.model)
        
        # Input tensor (batch=2, seq_len=10, hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = mla(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_pytorch_mamba_block(self):
        """Xác minh block Mamba-2 thuần PyTorch trả về shape và aux_loss đúng."""
        block = PyTorchMambaBlock(hidden_size=32)
        x = torch.randn(2, 8, 32)
        out, aux = block(x)
        self.assertEqual(out.shape, (2, 8, 32))
        self.assertEqual(aux.item(), 0.0)

    def test_pondernet_forward_and_loss(self):
        """Xác minh unroll động PonderNet và Halting Loss."""
        cfg = tiny_config()
        cfg.model.use_pondernet = True
        cfg.model.pondernet_prior_p = 0.3
        cfg.model.tokenizer_type = "vs_bpe"
        
        model = BigramModel(cfg.model)
        
        token_ids = torch.randint(0, 100, (2, 8))
        # Forward unroll PonderNet
        outputs = model(token_ids, num_recurrence=6)
        
        self.assertIn("logits", outputs)
        self.assertIn("halting_loss", outputs)
        self.assertTrue(outputs["halting_loss"].item() > 0.0)
        
        # Thử tính tổng loss
        targets = torch.randint(0, 100, (2, 8))
        loss_dict = compute_total_loss(outputs, targets, cfg.model)
        
        self.assertIn("total", loss_dict)
        self.assertIn("halting", loss_dict)
        self.assertTrue(loss_dict["total"].item() > 0.0)
        
        # Kiểm tra lan truyền ngược gradient
        loss_dict["total"].backward()
        # Đảm bảo gradient truyền qua được ponder_halt_head
        self.assertIsNotNone(model.ponder_halt_head.weight.grad)

    def test_bigram_v2_1_8b_a6000_config(self):
        """Xác minh cấu hình siêu lõi 1.8B tối ưu A6000 hợp lệ."""
        cfg = bigram_v2_1_8b_a6000_config()
        self.assertEqual(cfg.model.vocab_size, 32000)
        self.assertEqual(cfg.model.hidden_size, 2048)
        self.assertTrue(cfg.model.use_mla)
        self.assertTrue(cfg.model.use_pondernet)
        self.assertTrue(cfg.model.use_mamba)
        self.assertTrue(cfg.model.use_moe)

def run_all():
    print("Đang chạy test_upgrades...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBigramV2Upgrades)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("test_upgrades: CÓ LỖI THẤT BẠI")
    print("test_upgrades: TẤT CẢ ĐỀU PASS\n")

if __name__ == "__main__":
    run_all()
