"""
prepare.py
==========
Chuyển văn bản thô (.txt) thành các file nhị phân mà PackedDataset đọc được.

Quy trình:
  1. Đọc từng dòng văn bản.
  2. Dùng BigramTokenizer mã hóa -> (token_ids, tone_ids).
  3. Nối tất cả lại, ghi ra:
       - <prefix>.tok.bin   (uint16)
       - <prefix>.tone.bin  (uint8)

Hàm này được gọi bởi script `scripts/prepare_data.py`.
"""

import numpy as np


def prepare_corpus(text_path: str, tokenizer, out_prefix: str,
                   add_special_per_line: bool = True) -> dict:
    """
    Mã hóa một file văn bản thành file nhị phân.

    text_path           : đường dẫn file .txt nguồn (mỗi dòng một đoạn).
    tokenizer           : BigramTokenizer đã train.
    out_prefix          : tiền tố đường dẫn output. Sẽ tạo:
                            <out_prefix>.tok.bin và <out_prefix>.tone.bin
    add_special_per_line: nếu True, mỗi dòng được bọc <bos>...<eos>.
                          Điều này giúp model học ranh giới văn bản.

    Trả về dict thống kê (số dòng, số token).
    """
    all_tokens = []
    all_tones = []
    n_lines = 0

    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            tok, tone = tokenizer.encode(line, add_special=add_special_per_line)
            if tone is None:
                tone = [0] * len(tok)
            all_tokens.extend(tok)
            all_tones.extend(tone)
            n_lines += 1

    # Chuyển sang numpy với kiểu gọn nhẹ.
    # uint16 đủ cho vocab tới 65535; uint8 đủ cho 8 thanh điệu.
    tokens_arr = np.array(all_tokens, dtype=np.uint16)
    tones_arr = np.array(all_tones, dtype=np.uint8)

    # Kiểm tra tràn số: nếu vocab > 65535 thì uint16 không đủ.
    assert tokenizer.vocab_size <= 65535, \
        "Vocab quá lớn cho uint16 — cần đổi dtype sang uint32"

    tok_path = out_prefix + ".tok.bin"
    tone_path = out_prefix + ".tone.bin"
    tokens_arr.tofile(tok_path)
    tones_arr.tofile(tone_path)

    return {
        "n_lines": n_lines,
        "n_tokens": len(all_tokens),
        "tok_file": tok_path,
        "tone_file": tone_path,
    }
