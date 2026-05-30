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
    # Kiểm tra tránh NaN loss khi toàn bộ batch đều bị ignore
    active_mask = (targets != ignore_index)
    if not active_mask.any():
        return 0.0 * logits.sum()

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
                       tool_router_targets: torch.Tensor = None,
                       tool_name_targets: torch.Tensor = None,
                       verifier_targets: torch.Tensor = None,
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

    # 5) Tool loss — router toàn chuỗi, tool name chỉ tại vị trí tool_call.
    tool = torch.tensor(0.0, device=lm.device)
    router_loss = torch.tensor(0.0, device=lm.device)
    name_loss = torch.tensor(0.0, device=lm.device)
    if tool_router_targets is not None and "tool_router_logits" in outputs:
        b, s, c = outputs["tool_router_logits"].shape
        router_loss = F.cross_entropy(
            outputs["tool_router_logits"].reshape(b * s, c),
            tool_router_targets.reshape(b * s),
            ignore_index=ignore_index,
        )
        tool = router_loss
    if tool_name_targets is not None and "tool_name_logits" in outputs:
        name_targets = tool_name_targets.clone()
        if tool_router_targets is not None:
            name_targets = name_targets.masked_fill(
                tool_router_targets != 1, ignore_index)
        b, s, n = outputs["tool_name_logits"].shape
        valid = name_targets != ignore_index
        if valid.any():
            name_loss = F.cross_entropy(
                outputs["tool_name_logits"].reshape(b * s, n),
                name_targets.reshape(b * s),
                ignore_index=ignore_index,
            )
            tool = tool + name_loss

    # 6) Verifier loss — nhị phân claim/source supported score.
    verifier = torch.tensor(0.0, device=lm.device)
    if verifier_targets is not None and "verifier_logits" in outputs:
        v_targets = verifier_targets.float()
        valid = verifier_targets != ignore_index
        if valid.any():
            raw = F.binary_cross_entropy_with_logits(
                outputs["verifier_logits"], v_targets, reduction="none"
            )
            verifier = raw.masked_select(valid).mean()

    # 6.5) Halting loss — PonderNet halting loss.
    halting = outputs.get("halting_loss", torch.tensor(0.0, device=lm.device))
    if not torch.is_tensor(halting):
        halting = torch.tensor(float(halting), device=lm.device)

    # 7) Tổng có trọng số.
    total = (lm
             + config.abstention_loss_coef * abst
             + config.tone_loss_coef * tone
             + config.tool_loss_coef * tool
             + config.verifier_loss_coef * verifier
             + aux
             + halting)

    return {
        "total": total,
        "lm": lm.detach(),
        "abstention": abst.detach() if torch.is_tensor(abst) else abst,
        "tone": tone.detach() if torch.is_tensor(tone) else tone,
        "tool": tool.detach() if torch.is_tensor(tool) else tool,
        "tool_router": router_loss.detach() if torch.is_tensor(router_loss) else router_loss,
        "tool_name": name_loss.detach() if torch.is_tensor(name_loss) else name_loss,
        "verifier": verifier.detach() if torch.is_tensor(verifier) else verifier,
        "aux": aux.detach() if torch.is_tensor(aux) else aux,
        "halting": halting.detach() if torch.is_tensor(halting) else halting,
    }
