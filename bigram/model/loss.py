"""
loss.py
=======
Hàm mất mát (loss) tổng hợp cho Bigram.

Loss tổng = LM loss  +  abstention_coef * abstention loss  +  aux_loss (MoE)

  - LM loss        : cross-entropy dự đoán token tiếp theo (loss ngôn ngữ cơ bản).
  - Abstention loss : binary cross-entropy dạy model biết khi nào NÊN từ chối.
                      Đây là trụ cột chống hallucination ở mức huấn luyện.
  - aux_loss        : đã tính sẵn trong MoE để cân bằng tải expert.
"""

import torch
import torch.nn.functional as F


def language_modeling_loss(logits: torch.Tensor,
                           targets: torch.Tensor,
                           ignore_index: int = -100) -> torch.Tensor:
    """
    Cross-entropy loss cho bài toán dự đoán token tiếp theo.

    logits : (batch, seq_len, vocab_size) — điểm số model dự đoán.
    targets: (batch, seq_len)             — token đúng (đã dịch trái 1 vị trí).

    ignore_index: các vị trí có giá trị này (ví dụ padding) sẽ bị bỏ qua.
    """
    b, s, v = logits.shape
    # F.cross_entropy nhận (N, C) và (N,). Gộp batch & seq lại.
    loss = F.cross_entropy(
        logits.reshape(b * s, v),
        targets.reshape(b * s),
        ignore_index=ignore_index,
    )
    return loss


def abstention_loss(abstention_logits: torch.Tensor,
                    abstention_targets: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
    """
    Binary cross-entropy cho abstention head.

    abstention_logits : (batch, seq_len) — điểm "nên từ chối" (chưa qua sigmoid).
    abstention_targets: (batch, seq_len) — nhãn 1.0 = NÊN từ chối, 0.0 = nên trả lời.
    mask              : (batch, seq_len) — 1.0 ở vị trí cần tính loss, 0.0 bỏ qua.

    Dùng `with_logits` để ổn định số học (gộp sigmoid + BCE).
    """
    loss = F.binary_cross_entropy_with_logits(
        abstention_logits, abstention_targets, reduction="none"
    )
    if mask is not None:
        # Chỉ tính trung bình trên các vị trí được mask.
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        loss = loss.mean()
    return loss


def compute_total_loss(outputs: dict,
                       targets: torch.Tensor,
                       config,
                       abstention_targets: torch.Tensor = None,
                       abstention_mask: torch.Tensor = None,
                       tone_targets: torch.Tensor = None,
                       ignore_index: int = -100) -> dict:
    """
    Gộp tất cả thành phần loss lại.

    Tham số:
      outputs            : dict trả về từ BigramModel.forward().
      targets            : (batch, seq_len) — token đích cho LM loss.
      config             : ModelConfig (lấy các hệ số coef).
      abstention_targets : nhãn cho abstention head (tùy chọn, dùng ở giai
                           đoạn calibration).
      abstention_mask    : mask cho abstention loss.

    Trả về dict gồm: total, lm, abstention, aux (tiện cho việc log).
    """
    # 1) LM loss — luôn có.
    lm = language_modeling_loss(outputs["logits"], targets, ignore_index)

    # 2) MoE aux loss — đã tính sẵn trong forward.
    aux = outputs.get("aux_loss", torch.tensor(0.0, device=lm.device))
    # aux có thể là float 0.0 nếu không dùng MoE -> ép về tensor.
    if not torch.is_tensor(aux):
        aux = torch.tensor(float(aux), device=lm.device)

    # 3) Abstention loss — chỉ tính khi có nhãn (giai đoạn calibration).
    abst = torch.tensor(0.0, device=lm.device)
    if abstention_targets is not None and "abstention_logits" in outputs:
        abst = abstention_loss(
            outputs["abstention_logits"], abstention_targets, abstention_mask
        )

    # 4) Tone loss — cross-entropy cho luồng thanh điệu.
    # Tính khi model có tone head và có nhãn thanh điệu đích. Đây là loss
    # giúp văn bản sinh ra giữ đúng dấu thanh tiếng Việt.
    tone = torch.tensor(0.0, device=lm.device)
    if tone_targets is not None and "tone_logits" in outputs:
        tone = language_modeling_loss(
            outputs["tone_logits"], tone_targets, ignore_index
        )

    # 5) Tổng có trọng số.
    total = (lm
             + config.abstention_loss_coef * abst
             + config.tone_loss_coef * tone
             + aux)

    return {
        "total": total,
        "lm": lm.detach(),
        "abstention": abst.detach() if torch.is_tensor(abst) else abst,
        "tone": tone.detach() if torch.is_tensor(tone) else tone,
        "aux": aux.detach() if torch.is_tensor(aux) else aux,
    }
