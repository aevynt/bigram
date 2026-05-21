"""
postprocess.py
==============
Normalize lỗi dấu tiếng Việt trong output của Model A
trước khi parse JSON tool_call.

Áp dụng sau khi model sinh output, trước khi parse <tool_call>.
"""

TONE_FIXES = {
    # Lỗi phổ biến nhất
    "gía": "giá",
    "Gía": "Giá",
    "tỷ gía": "tỷ giá",
    "Tỷ gía": "Tỷ giá",
    "gía vàng": "giá vàng",
    "gía vang": "giá vàng",
    "gía USD": "giá USD",
    "gía xăng": "giá xăng",
    # Lỗi dấu khác
    "qủa": "quả",
    "Qủa": "Quả",
    "kết qủa": "kết quả",
    "Kết qủa": "Kết quả",
    "đươc": "được",
    "Đươc": "Được",
    "tưong": "tương",
    "Tưong": "Tương",
    "tường đương": "tương đương",
    "yều": "yêu",
    "Yều": "Yêu",
    "hốm": "hôm",
    "Hốm": "Hôm",
}


def fix_tone(text: str) -> str:
    """Normalize lỗi dấu tiếng Việt trong một chuỗi."""
    for wrong, right in TONE_FIXES.items():
        text = text.replace(wrong, right)
    return text


def fix_output(model_output: str) -> str:
    """
    Normalize toàn bộ output của Model A.
    Áp dụng trước khi parse <tool_call> hay hiển thị.
    """
    return fix_tone(model_output)
