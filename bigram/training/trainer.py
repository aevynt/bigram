"""
trainer.py
==========
Vòng lặp huấn luyện (training loop) chính của Bigram.

Lớp `Trainer` lo toàn bộ: tích lũy gradient, clip gradient, mixed precision,
cập nhật learning rate, đánh giá định kỳ, lưu/nạp checkpoint, ghi log.

Trainer dùng được cho mọi giai đoạn của pipeline (pretrain / midtrain / sft);
khác biệt giữa các giai đoạn chủ yếu nằm ở DỮ LIỆU và CONFIG, không phải ở
bản thân vòng lặp.
"""

import os
import time
import math
import torch
from torch.utils.data import DataLoader

from ..model import compute_total_loss
from .optim import build_optimizer, get_lr, apply_lr


class Trainer:
    """Quản lý quá trình huấn luyện Bigram."""

    def __init__(self, model, config, train_dataset, val_dataset=None):
        """
        model         : một BigramModel.
        config        : BigramConfig (cả model + train + data).
        train_dataset : Dataset dùng để train.
        val_dataset   : Dataset để đánh giá (tùy chọn).
        """
        self.config = config
        self.tcfg = config.train
        self.mcfg = config.model

        # --- Chọn thiết bị ---
        if self.tcfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.tcfg.device
        self.model = model.to(self.device)
        if hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = bool(self.tcfg.gradient_checkpointing)

        # --- DataLoader ---
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.tcfg.batch_size,
            shuffle=True, drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.tcfg.batch_size,
                shuffle=False, drop_last=True,
            )

        # --- Optimizer ---
        self.optimizer = build_optimizer(model, self.tcfg)

        # --- Mixed precision (chỉ bật khi có GPU) ---
        self.use_amp = self.tcfg.use_amp and self.device == "cuda"
        # GradScaler giúp tránh underflow gradient khi train fp16.
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        # bf16 ổn định hơn fp16; dùng bf16 nếu GPU hỗ trợ.
        self.amp_dtype = torch.bfloat16 if (
            self.device == "cuda" and torch.cuda.is_bf16_supported()
        ) else torch.float16

        self.step = 0
        os.makedirs(self.tcfg.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _move_batch(self, batch):
        """Chuyển một batch lên thiết bị tính toán."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward_loss(self, batch):
        """Chạy forward + tính loss cho một batch. Trả về dict loss."""
        out = self.model(batch["token_ids"], batch["tone_ids"])
        losses = compute_total_loss(
            out, batch["targets"], self.mcfg,
            abstention_targets=batch.get("abstention_targets"),
            abstention_mask=batch.get("abstention_mask"),
            tone_targets=batch.get("tone_targets"),
            tool_router_targets=batch.get("tool_router_targets"),
            tool_name_targets=batch.get("tool_name_targets"),
            verifier_targets=batch.get("verifier_targets"),
        )
        return losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> dict:
        """Đánh giá model trên tập validation. Trả về loss trung bình + perplexity."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_lm, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            batch = self._move_batch(batch)
            losses = self._forward_loss(batch)
            total_lm += losses["lm"].item()
            n += 1
        self.model.train()
        avg_lm = total_lm / max(1, n)
        return {
            "val_lm": avg_lm,
            # Perplexity = exp(cross-entropy) — thước đo trực giác hơn.
            "val_ppl": math.exp(min(avg_lm, 20)),  # clamp tránh tràn số.
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str = None):
        """Lưu checkpoint (model + optimizer + bước hiện tại + config)."""
        if name is None:
            name = f"ckpt_step{self.step}.pt"
        path = os.path.join(self.tcfg.out_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "config": {
                "model": vars(self.mcfg),
                "train": vars(self.tcfg),
            },
        }, path)
        return path

    def load_checkpoint(self, path: str):
        """Nạp lại checkpoint để train tiếp."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)

    # ------------------------------------------------------------------
    def train(self):
        """
        Chạy vòng lặp huấn luyện chính cho tới khi đạt `max_steps`.

        Cơ chế tích lũy gradient (gradient accumulation):
        Để có "batch hiệu dụng" lớn mà không tốn bộ nhớ, ta chia một batch lớn
        thành `grad_accum_steps` batch nhỏ, cộng dồn gradient, rồi mới cập nhật.
        """
        self.model.train()
        accum = self.tcfg.grad_accum_steps
        t0 = time.time()
        # Tạo iterator vô hạn từ train_loader.
        data_iter = iter(self.train_loader)

        running_loss = 0.0
        while self.step < self.tcfg.max_steps:
            # --- Cập nhật learning rate cho bước này ---
            lr = get_lr(self.step, self.tcfg)
            apply_lr(self.optimizer, lr)

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            # --- Tích lũy gradient qua `accum` micro-batch ---
            for _ in range(accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Hết dữ liệu -> tạo lại iterator (lặp lại epoch mới).
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                batch = self._move_batch(batch)

                # Forward trong autocast (mixed precision nếu bật).
                with torch.amp.autocast(device_type=self.device.split(":")[0],
                                        dtype=self.amp_dtype,
                                        enabled=self.use_amp):
                    losses = self._forward_loss(batch)
                    # Chia loss cho accum vì ta cộng dồn gradient.
                    loss = losses["total"] / accum

                # Backward — scaler nhân loss lên để tránh underflow fp16.
                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # --- Clip gradient (đặc biệt quan trọng cho recurrent) ---
            # unscale trước khi clip để clip đúng độ lớn thật.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tcfg.grad_clip)

            # --- Cập nhật trọng số ---
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += step_loss
            self.step += 1

            # --- Logging ---
            if self.step % self.tcfg.log_interval == 0:
                avg = running_loss / self.tcfg.log_interval
                dt = time.time() - t0
                ips = self.tcfg.log_interval / dt  # iteration/giây.
                print(f"step {self.step:6d} | loss {avg:.4f} | "
                      f"lr {lr:.2e} | {ips:.2f} it/s")
                running_loss = 0.0
                t0 = time.time()

            # --- Đánh giá định kỳ ---
            if (self.val_loader is not None
                    and self.step % self.tcfg.eval_interval == 0):
                metrics = self.evaluate()
                print(f"  >> eval @ step {self.step}: "
                      f"val_lm {metrics['val_lm']:.4f} | "
                      f"val_ppl {metrics['val_ppl']:.2f}")

            # --- Lưu checkpoint định kỳ ---
            if self.step % self.tcfg.save_interval == 0:
                path = self.save_checkpoint()
                print(f"  >> đã lưu checkpoint: {path}")

        # Lưu checkpoint cuối cùng.
        final = self.save_checkpoint("ckpt_final.pt")
        print(f"Hoàn tất huấn luyện. Checkpoint cuối: {final}")
        return final
