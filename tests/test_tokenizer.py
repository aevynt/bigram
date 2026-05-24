"""
test_tokenizer.py
=================
Bộ kiểm thử cho tokenizer tiếng Việt của Bigram.

Chạy bằng:  pytest tests/test_tokenizer.py -v
Hoặc:       python tests/test_tokenizer.py

Bao phủ:
  - Tách / ghép thanh điệu (split_tone / merge_tone).
  - Giữ đúng nguyên âm đặc biệt (ă, â, ê, ô, ơ, ư, đ).
  - Train BPE và encode/decode round-trip.
  - Token đặc biệt (<bos>, <eos>) hoạt động đúng.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram.tokenizer import (
    BigramTokenizer, split_tone, merge_tone, apply_tone_to_syllable,
    tones_to_ids, ids_to_tones, TONE_NAMES,
)

# Văn bản mẫu để train tokenizer trong test.
_SAMPLE_TEXT = """Tiếng Việt là ngôn ngữ chính thức của Việt Nam.
Hà Nội là thủ đô nghìn năm văn hiến của đất nước.
Học sinh chăm chỉ học tập mỗi ngày để xây dựng tương lai.
Trí tuệ nhân tạo đang thay đổi cách con người làm việc.
Những cánh đồng lúa chín vàng trải dài tới chân trời.
Dòng sông quê hương êm đềm chảy qua bao thế hệ.
""" * 30


def _get_tokenizer(tmp_dir=None):
    """Train một tokenizer nhỏ cho mục đích test."""
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, "_test_tok_corpus.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_SAMPLE_TEXT)
    return BigramTokenizer.train([path], vocab_size=500, min_frequency=1)


def test_split_tone_basic():
    """Tách thanh điệu cơ bản: 'má' -> ('ma', [..., 'sac'])."""
    base, tones = split_tone("má")
    assert base == "ma"
    assert "sac" in tones
    print("  [OK] test_split_tone_basic")


def test_split_tone_keeps_vowel_modifiers():
    """Phải GIỮ ư/ê/ô/đ — chỉ tách dấu THANH."""
    base, _ = split_tone("những")
    assert base == "nhưng", f"Mong 'nhưng', nhận '{base}'"
    base2, _ = split_tone("đường")
    assert base2 == "đương", f"Mong 'đương', nhận '{base2}'"
    print("  [OK] test_split_tone_keeps_vowel_modifiers")


def test_merge_tone_roundtrip():
    """split rồi merge phải khôi phục nguyên văn."""
    words = ["những", "lướt", "Nguyễn", "tiếng Việt", "đường phố"]
    for w in words:
        base, tones = split_tone(w)
        assert merge_tone(base, tones) == w, f"Round-trip hỏng ở '{w}'"
    print("  [OK] test_merge_tone_roundtrip")


def test_tone_id_conversion():
    """Chuyển thanh điệu sang id và ngược lại."""
    _, tones = split_tone("tiếng Việt")
    ids = tones_to_ids(tones)
    assert all(isinstance(i, int) for i in ids)
    back = ids_to_tones(ids)
    assert back == tones
    print("  [OK] test_tone_id_conversion")


def test_apply_tone_to_syllable():
    """Đặt dấu thanh đúng vị trí theo chính tả."""
    # 'tiêng' + sắc -> 'tiếng' (dấu vào ê).
    assert apply_tone_to_syllable("tiêng", "sac") == "tiếng"
    # 'hoc' + nặng -> 'học'.
    assert apply_tone_to_syllable("hoc", "nang") == "học"
    print("  [OK] test_apply_tone_to_syllable")


def test_tokenizer_train():
    """Train tokenizer phải tạo được vocab không rỗng."""
    tok = _get_tokenizer()
    assert tok.vocab_size > 4  # ít nhất phải nhiều hơn 4 token đặc biệt.
    print(f"  [OK] test_tokenizer_train (vocab={tok.vocab_size})")


def test_tokenizer_roundtrip():
    """encode rồi decode phải khôi phục văn bản (có dấu)."""
    tok = _get_tokenizer()
    texts = ["Tiếng Việt mến yêu", "Hà Nội nghìn năm",
             "những cánh đồng lúa chín", "Trí tuệ nhân tạo"]
    for text in texts:
        ids, tones = tok.encode(text, add_special=True)
        decoded = tok.decode(ids, tones)
        assert decoded == text, f"Round-trip hỏng:\n  gốc: {text}\n  ra : {decoded}"
    print("  [OK] test_tokenizer_roundtrip")


def test_tokenizer_special_tokens():
    """<bos>/<eos> phải được thêm vào và loại bỏ đúng."""
    tok = _get_tokenizer()
    ids, tones = tok.encode("Hà Nội", add_special=True)
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    assert ids[0] == bos and ids[-1] == eos
    # decode phải bỏ token đặc biệt.
    decoded = tok.decode(ids, tones)
    assert "<bos>" not in decoded and "<eos>" not in decoded
    print("  [OK] test_tokenizer_special_tokens")


def test_tokenizer_save_load(tmp_path=None):
    """Lưu rồi nạp lại tokenizer phải cho kết quả y hệt."""
    if tmp_path is None:
        tmp_path = tempfile.gettempdir()
    tok = _get_tokenizer()
    path = os.path.join(tmp_path, "_test_tok_save.json")
    tok.save(path)
    tok2 = BigramTokenizer.load(path)
    assert tok.vocab_size == tok2.vocab_size
    ids1, _ = tok.encode("Việt Nam")
    ids2, _ = tok2.encode("Việt Nam")
    assert ids1 == ids2
    print("  [OK] test_tokenizer_save_load")


def run_all():
    print("Đang chạy test_tokenizer...")
    test_split_tone_basic()
    test_split_tone_keeps_vowel_modifiers()
    test_merge_tone_roundtrip()
    test_tone_id_conversion()
    test_apply_tone_to_syllable()
    test_tokenizer_train()
    test_tokenizer_roundtrip()
    test_tokenizer_special_tokens()
    test_tokenizer_save_load()
    print("test_tokenizer: TẤT CẢ ĐỀU PASS\n")


if __name__ == "__main__":
    run_all()
