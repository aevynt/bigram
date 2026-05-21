"""
helpers.py
==========
Các hàm tiện ích dùng chung trong codebase Bigram.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Cố định mọi nguồn ngẫu nhiên để thí nghiệm tái lập được.
    Gọi hàm này ở đầu mỗi script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> dict:
    """
    Đếm số tham số của model, chia theo nhóm (prelude / recurrent / coda / khác).
    Hữu ích để hiểu model "nặng" ở đâu.
    """
    groups = {"prelude": 0, "recurrent": 0, "coda": 0,
              "embedding": 0, "head": 0, "khác": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("prelude"):
            groups["prelude"] += n
        elif name.startswith("recurrent"):
            groups["recurrent"] += n
        elif name.startswith("coda"):
            groups["coda"] += n
        elif "embedding" in name:
            groups["embedding"] += n
        elif "head" in name:
            groups["head"] += n
        else:
            groups["khác"] += n
    groups["TỔNG"] = sum(v for k, v in groups.items() if k != "TỔNG")
    return groups


def format_number(n: int) -> str:
    """Định dạng số lớn cho dễ đọc: 1500000 -> '1.5M'."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def estimate_effective_depth(config) -> int:
    """
    Tính "độ sâu hiệu dụng" của model khi lặp khối recurrent.

    Một transformer thường có độ sâu = số layer. Bigram, khi lặp khối lõi r
    lần, có độ sâu hiệu dụng = lP + lR*r + lC. Đây là lý do model ít tham số
    nhưng vẫn "suy luận sâu".
    """
    r = int(round(config.mean_recurrence))
    return (config.n_prelude_layers
            + config.n_recurrent_layers * r
            + config.n_coda_layers)
