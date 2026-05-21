"""
tonal.py
========
Bộ tách thanh điệu tiếng Việt — phần "tiền xử lý" cho tokenizer của Bigram.

Ý tưởng cốt lõi (xem PHILOSOPHY.md):
Tiếng Việt có 6 thanh điệu. Một tokenizer thông thường coi "ma / má / mà /
mả / mã / mạ" gần như 6 token độc lập, dù chúng chia sẻ cùng âm gốc "ma".
Điều này (a) làm phình từ điển và (b) khiến model khó học quy luật âm vị.

Giải pháp Bigram: tách mỗi ký tự có dấu thành (ký_tự_gốc, thanh_điệu).
  "những"  ->  âm gốc "nhưng"  +  thanh "huyền"
  "lướt"   ->  âm gốc "luơt"   +  thanh "sắc"   (lưu ý: ơ vẫn giữ, chỉ bỏ dấu thanh)

Lưu ý quan trọng: ta CHỈ tách dấu THANH (sắc/huyền/hỏi/ngã/nặng), KHÔNG
tách dấu phụ tạo nguyên âm (ă, â, ê, ô, ơ, ư, đ). Vì ă/â/ê... là âm vị
riêng biệt, còn dấu thanh mới là thứ lặp lại có quy luật.
"""

import unicodedata

# 6 thanh điệu tiếng Việt. Index 0 dành cho "không có thanh" (token <none>).
TONE_NAMES = ["<none>", "ngang", "huyen", "sac", "hoi", "nga", "nang"]
TONE_TO_ID = {name: i for i, name in enumerate(TONE_NAMES)}

# Mã Unicode combining cho từng dấu thanh (ký tự kết hợp - combining mark).
# Khi chuẩn hóa NFD, một chữ có dấu tách thành (chữ gốc + các combining mark).
COMBINING_TO_TONE = {
    "\u0300": "huyen",  # huyền (grave accent)
    "\u0301": "sac",    # sắc   (acute accent)
    "\u0303": "nga",    # ngã   (tilde)
    "\u0309": "hoi",    # hỏi   (hook above)
    "\u0323": "nang",   # nặng  (dot below)
}

# Các combining mark TẠO NGUYÊN ÂM (không phải thanh) — phải GIỮ LẠI.
#   \u0302 = dấu mũ (â, ê, ô)
#   \u0306 = dấu á trăng (ă)
#   \u031b = dấu móc (ơ, ư)
VOWEL_MODIFIERS = {"\u0302", "\u0306", "\u031b"}


def split_tone(text: str):
    """
    Tách một chuỗi thành (chuỗi_âm_gốc, danh_sách_thanh_điệu).

    Trả về:
      base_text : chuỗi đã bỏ dấu thanh (nhưng giữ ă, â, ê, ô, ơ, ư, đ).
      tones     : list cùng độ dài base_text; mỗi phần tử là tên thanh điệu
                  của ký tự tương ứng ("<none>" nếu ký tự đó không mang thanh).

    Ví dụ:
      split_tone("má")  -> ("ma", ["<none>", "sac"])  (thanh gắn với nguyên âm)
    """
    # Chuẩn hóa NFD: tách chữ có dấu thành chữ gốc + các combining mark.
    decomposed = unicodedata.normalize("NFD", text)

    base_chars = []   # các ký tự của chuỗi âm gốc.
    tones = []        # thanh điệu tương ứng từng ký tự.

    for ch in decomposed:
        if ch in COMBINING_TO_TONE:
            # Đây là dấu thanh -> gán cho ký tự gần nhất vừa thêm.
            if base_chars:
                tones[-1] = COMBINING_TO_TONE[ch]
            # Không thêm ch vào base_chars (dấu thanh bị "tách ra").
        elif ch in VOWEL_MODIFIERS:
            # Dấu tạo nguyên âm -> GỮ LẠI, gắn ngược vào ký tự trước.
            if base_chars:
                base_chars[-1] = unicodedata.normalize(
                    "NFC", base_chars[-1] + ch)
        else:
            # Ký tự bình thường.
            base_chars.append(ch)
            tones.append("<none>")

    base_text = "".join(base_chars)
    return base_text, tones


def merge_tone(base_text: str, tones: list) -> str:
    """
    Phép NGƯỢC của split_tone: ghép âm gốc + thanh điệu lại thành chữ có dấu.

    Dùng khi giải mã (decode) output của model trở lại văn bản đọc được.
    """
    # Tên thanh -> combining mark tương ứng.
    tone_to_combining = {v: k for k, v in COMBINING_TO_TONE.items()}

    out = []
    for ch, tone in zip(base_text, tones):
        if tone in tone_to_combining:
            # Ghép combining mark vào rồi chuẩn hóa NFC (gộp thành 1 ký tự).
            combined = unicodedata.normalize("NFC", ch + tone_to_combining[tone])
            out.append(combined)
        else:
            out.append(ch)
    return "".join(out)


def apply_tone_to_syllable(syllable: str, tone_name: str) -> str:
    """
    Đặt dấu thanh vào một âm tiết (syllable) theo quy tắc chính tả tiếng Việt.

    Quy tắc (theo "kiểu mới", áp dụng cho đa số trường hợp):
      1. Nếu âm tiết chỉ có một nguyên âm -> đặt dấu vào nguyên âm đó.
      2. Nếu có cụm nguyên âm:
         a. Cụm chứa ê/ơ -> dấu LUÔN vào ê hoặc ơ (đây là âm chính).
            Ví dụ: "ươc" -> dấu vào ơ ("nước"); "iên" -> dấu vào ê ("tiến").
         b. Cụm chứa â/ă/ô -> dấu vào â/ă/ô.
         c. Còn lại: nếu có âm cuối (phụ âm sau cụm nguyên âm) -> dấu vào
            nguyên âm CUỐI của cụm; nếu không -> dấu vào nguyên âm ÁP CHÓT.
            Ví dụ: "oan" -> dấu vào a ("toàn"); "ua" -> dấu vào u ("của").

    Hàm này dùng cho việc DECODE đẹp; lúc train model không cần nó.
    """
    if tone_name == "<none>" or tone_name not in COMBINING_TO_TONE.values():
        return syllable

    vowels = "aăâeêioôơuưy"
    # Tìm vị trí tất cả nguyên âm trong âm tiết.
    vowel_pos = [i for i, c in enumerate(syllable) if c.lower() in vowels]
    if not vowel_pos:
        return syllable  # không có nguyên âm -> không đặt được dấu.

    target = None

    # --- Bước 1: âm tiết chỉ một nguyên âm ---
    if len(vowel_pos) == 1:
        target = vowel_pos[0]
    else:
        # --- Bước 2a: ưu tiên tuyệt đối cho ê và ơ ---
        # Trong các cụm như "ươ", "iê", "yê", ê/ơ luôn là âm chính mang dấu.
        for i in vowel_pos:
            if syllable[i].lower() in "êơ":
                target = i
                break

        # --- Bước 2b: kế đến là â, ă, ô ---
        if target is None:
            for i in vowel_pos:
                if syllable[i].lower() in "âăô":
                    target = i
                    break

        # --- Bước 2c: cụm nguyên âm thường — xét âm cuối ---
        if target is None:
            last_vowel = vowel_pos[-1]
            has_final = last_vowel < len(syllable) - 1
            if has_final:
                # Có phụ âm cuối -> dấu vào nguyên âm cuối của cụm.
                target = last_vowel
            else:
                # Không âm cuối -> đặt vào nguyên âm áp chót.
                target = vowel_pos[-2]

    chars = list(syllable)
    chars[target] = merge_tone(chars[target], [tone_name])
    return "".join(chars)


def tones_to_ids(tones: list) -> list:
    """Chuyển danh sách tên thanh điệu thành danh sách id số."""
    return [TONE_TO_ID.get(t, 0) for t in tones]


def ids_to_tones(ids: list) -> list:
    """Chuyển danh sách id số trở lại tên thanh điệu."""
    return [TONE_NAMES[i] if 0 <= i < len(TONE_NAMES) else "<none>" for i in ids]
