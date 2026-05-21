"""
dpo.py
======
Direct Preference Optimization (DPO) — giai đoạn 3b của pipeline alignment.

Vì sao DPO thay vì RL/PPO:
RLHF cổ điển cần (1) train một reward model riêng, rồi (2) dùng PPO — một
thuật toán RL nổi tiếng khó tinh chỉnh và dễ sụp đổ. DPO bỏ cả hai: nó biến
bài toán về một dạng phân loại đơn giản, ổn định như học có giám sát thông
thường. Cho usecase "AI chính xác cho tổ chức" của Bigram, sự ỔN ĐỊNH này
quan trọng hơn mức trần hiệu năng lý thuyết của PPO.

Ý tưởng cốt lõi:
Cho một prompt và hai câu trả lời — y_chosen (người chấm thích hơn) và
y_rejected (kém hơn) — DPO huấn luyện model sao cho nó tăng xác suất tương
đối của y_chosen so với y_rejected, ĐỒNG THỜI không trôi quá xa khỏi một
"model tham chiếu" (reference policy) bị đông cứng.

Công thức loss (Rafailov et al. 2023):

    L = -log σ( β · [ (logπ_θ(y_w) - logπ_ref(y_w))
                     - (logπ_θ(y_l) - logπ_ref(y_l)) ] )

  trong đó:
    π_θ   : model đang train (policy).
    π_ref : model tham chiếu, đông cứng — thường là chính bản SFT.
    y_w   : câu trả lời được ưa thích (winner / chosen).
    y_l   : câu trả lời bị từ chối (loser / rejected).
    β     : hệ số kiểm soát mức độ bám sát π_ref (0.1–0.5 là khoảng tốt).
    σ     : hàm sigmoid.

Lưu ý: model tham chiếu KHÔNG được cập nhật — nó chỉ dùng để tính log-xác suất
làm mốc so sánh. Trong code, ta đông cứng nó bằng cách tắt requires_grad.
"""

import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor,
                     targets: torch.Tensor,
                     ignore_index: int = -100) -> torch.Tensor:
    """
    Tính tổng log-xác suất mà model gán cho một chuỗi đích.

    logits  : (batch, seq_len, vocab) — đầu ra model.
    targets : (batch, seq_len)        — token đích. Vị trí = ignore_index bị bỏ.

    Trả về: (batch,) — tổng log p(token) trên các vị trí không bị mask.

    Đây là "điểm số" mà DPO dùng: model thích một câu trả lời tới mức nào
    được đo bằng tổng log-xác suất nó gán cho các token của câu đó.
    """
    # log-softmax -> log xác suất cho mọi token trong từ điển.
    log_probs = F.log_softmax(logits, dim=-1)

    # Mask: vị trí nào thực sự tính (target khác ignore_index).
    mask = (targets != ignore_index)
    # Thay ignore_index bằng 0 để gather không lỗi chỉ số âm.
    safe_targets = targets.clone()
    safe_targets[~mask] = 0

    # gather: lấy log-xác suất của đúng token đích tại mỗi vị trí.
    # safe_targets.unsqueeze(-1): (b, s) -> (b, s, 1).
    token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

    # Chỉ cộng các vị trí hợp lệ.
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1)  # (batch,)


def dpo_loss(policy_chosen_logp: torch.Tensor,
             policy_rejected_logp: torch.Tensor,
             ref_chosen_logp: torch.Tensor,
             ref_rejected_logp: torch.Tensor,
             beta: float = 0.1) -> dict:
    """
    Tính DPO loss từ các log-xác suất đã tính sẵn.

    Bốn đầu vào, mỗi cái shape (batch,):
      policy_chosen_logp   : logπ_θ(y_w)   — model train, câu được thích.
      policy_rejected_logp : logπ_θ(y_l)   — model train, câu bị từ chối.
      ref_chosen_logp      : logπ_ref(y_w) — model tham chiếu, câu được thích.
      ref_rejected_logp    : logπ_ref(y_l) — model tham chiếu, câu bị từ chối.

    Trả về dict gồm: loss, và các "implicit reward" để theo dõi quá trình train.
    """
    # log tỉ lệ policy/reference cho từng câu trả lời.
    # Đây chính là "phần thưởng ngầm" mà DPO gán — model là reward model của
    # chính nó (tên paper: "Your Language Model is Secretly a Reward Model").
    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp

    # "logits" của bài toán phân loại: chênh lệch giữa hai log-tỉ lệ.
    logits = chosen_logratio - rejected_logratio

    # Loss = -log σ(β · logits). logsigmoid ổn định số học hơn log(sigmoid(.)).
    loss = -F.logsigmoid(beta * logits).mean()

    # Các đại lượng theo dõi (không ảnh hưởng gradient — chỉ để in log):
    #  - reward ngầm cho mỗi phía = β * log-tỉ lệ.
    #  - accuracy = tỉ lệ ví dụ mà chosen được chấm điểm cao hơn rejected.
    chosen_reward = (beta * chosen_logratio).detach()
    rejected_reward = (beta * rejected_logratio).detach()
    reward_accuracy = (chosen_reward > rejected_reward).float().mean()
    reward_margin = (chosen_reward - rejected_reward).mean()

    return {
        "loss": loss,
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "reward_accuracy": reward_accuracy,
        "reward_margin": reward_margin,
    }


def compute_dpo_loss_from_batch(policy_model, ref_model, batch,
                                beta: float = 0.1,
                                num_recurrence: int = None,
                                ignore_index: int = -100) -> dict:
    """
    Tính DPO loss trực tiếp từ một batch dữ liệu ưu tiên.

    batch là dict chứa (xem PreferenceDataset):
      chosen_token_ids, chosen_tone_ids, chosen_targets
      rejected_token_ids, rejected_tone_ids, rejected_targets

    policy_model : model đang train.
    ref_model    : model tham chiếu, đã đông cứng (no_grad).

    Hàm này chạy 4 lượt forward: policy/ref × chosen/rejected.
    """
    # --- Forward cho model policy (CÓ gradient) ---
    pol_chosen = policy_model(batch["chosen_token_ids"],
                              batch["chosen_tone_ids"],
                              num_recurrence=num_recurrence)
    pol_rejected = policy_model(batch["rejected_token_ids"],
                                batch["rejected_tone_ids"],
                                num_recurrence=num_recurrence)
    pol_chosen_logp = sequence_logprob(
        pol_chosen["logits"], batch["chosen_targets"], ignore_index)
    pol_rejected_logp = sequence_logprob(
        pol_rejected["logits"], batch["rejected_targets"], ignore_index)

    # --- Forward cho model tham chiếu (KHÔNG gradient — nó đông cứng) ---
    with torch.no_grad():
        ref_chosen = ref_model(batch["chosen_token_ids"],
                               batch["chosen_tone_ids"],
                               num_recurrence=num_recurrence)
        ref_rejected = ref_model(batch["rejected_token_ids"],
                                 batch["rejected_tone_ids"],
                                 num_recurrence=num_recurrence)
        ref_chosen_logp = sequence_logprob(
            ref_chosen["logits"], batch["chosen_targets"], ignore_index)
        ref_rejected_logp = sequence_logprob(
            ref_rejected["logits"], batch["rejected_targets"], ignore_index)

    return dpo_loss(pol_chosen_logp, pol_rejected_logp,
                    ref_chosen_logp, ref_rejected_logp, beta=beta)


# ----------------------------------------------------------------------
# DPOTrainer — vòng lặp huấn luyện cho giai đoạn DPO
# ----------------------------------------------------------------------
import copy
import os
import math
from torch.utils.data import DataLoader

from .optim import build_optimizer, get_lr, apply_lr


class DPOTrainer:
    """
    Quản lý quá trình huấn luyện DPO.

    Khác Trainer thường ở hai điểm:
      1. Có hai model: `policy` (được train) và `ref` (đông cứng làm mốc).
      2. Loss là DPO loss, không phải cross-entropy.

    Model tham chiếu được tạo bằng cách SAO CHÉP policy lúc khởi tạo rồi đông
    cứng — đúng tinh thần "ref policy là bản SFT" trong paper DPO.
    """

    def __init__(self, policy_model, config, train_dataset, beta=0.1):
        """
        policy_model  : BigramModel cần train (nên đã qua SFT).
        config        : BigramConfig.
        train_dataset : một PreferenceDataset.
        beta          : hệ số DPO (0.1–0.5).
        """
        self.config = config
        self.tcfg = config.train
        self.mcfg = config.model
        self.beta = beta

        if self.tcfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.tcfg.device

        # Policy: model được train.
        self.policy = policy_model.to(self.device)

        # Reference: bản sao đông cứng của policy tại thời điểm bắt đầu DPO.
        self.ref = copy.deepcopy(policy_model).to(self.device)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)  # đông cứng — không bao giờ cập nhật.

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.tcfg.batch_size,
            shuffle=True, drop_last=True,
        )
        self.optimizer = build_optimizer(self.policy, self.tcfg)
        self.step = 0
        os.makedirs(self.tcfg.out_dir, exist_ok=True)

    def _move_batch(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def save_checkpoint(self, name="dpo_final.pt"):
        """Lưu checkpoint của model policy (ref không cần lưu — tái tạo được)."""
        path = os.path.join(self.tcfg.out_dir, name)
        torch.save({
            "model": self.policy.state_dict(),
            "step": self.step,
            "config": {"model": vars(self.mcfg), "train": vars(self.tcfg)},
        }, path)
        return path

    def train(self):
        """Chạy vòng lặp huấn luyện DPO cho tới khi đạt max_steps."""
        self.policy.train()
        data_iter = iter(self.train_loader)
        running = {"loss": 0.0, "acc": 0.0, "margin": 0.0}

        while self.step < self.tcfg.max_steps:
            lr = get_lr(self.step, self.tcfg)
            apply_lr(self.optimizer, lr)
            self.optimizer.zero_grad(set_to_none=True)

            # Lấy một batch (lặp lại dataset nếu hết).
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            batch = self._move_batch(batch)

            # Tính DPO loss.
            out = compute_dpo_loss_from_batch(
                self.policy, self.ref, batch, beta=self.beta)
            loss = out["loss"]

            # Backward + clip + cập nhật.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.tcfg.grad_clip)
            self.optimizer.step()

            running["loss"] += loss.item()
            running["acc"] += out["reward_accuracy"].item()
            running["margin"] += out["reward_margin"].item()
            self.step += 1

            # Log định kỳ.
            if self.step % self.tcfg.log_interval == 0:
                n = self.tcfg.log_interval
                print(f"[DPO] step {self.step:6d} | "
                      f"loss {running['loss']/n:.4f} | "
                      f"reward_acc {running['acc']/n:.3f} | "
                      f"margin {running['margin']/n:.4f} | "
                      f"lr {lr:.2e}")
                running = {"loss": 0.0, "acc": 0.0, "margin": 0.0}

            if self.step % self.tcfg.save_interval == 0:
                p = self.save_checkpoint(f"dpo_step{self.step}.pt")
                print(f"  >> đã lưu: {p}")

        final = self.save_checkpoint("dpo_final.pt")
        print(f"Hoàn tất DPO. Checkpoint: {final}")
        return final
