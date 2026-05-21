"""
bpe.py
======
BigramTokenizer — tokenizer hoàn chỉnh của Bigram.

Kết hợp hai tầng:
  1. Tách thanh điệu (tonal.py): văn bản -> (âm gốc, danh sách thanh điệu).
  2. BPE trên âm gốc: âm gốc -> danh sách token id.

Kết quả mã hóa của một câu gồm HAI dãy số song song:
  - token_ids : id BPE của âm gốc.
  - tone_ids  : id thanh điệu, căn theo từng token.

Model Bigram nhận cả hai dãy này (xem BigramModel.embed).

LƯU Ý về việc căn thanh điệu theo token:
BPE gộp nhiều ký tự thành một token, nhưng thanh điệu gắn với KÝ TỰ. Để đơn
giản và ổn định, ta gán cho mỗi token thanh điệu của ký tự MANG THANH đầu
tiên bên trong token đó (đa số token chỉ chứa tối đa một nguyên âm có thanh).
Đây là một xấp xỉ thực dụng — đủ tốt cho mục tiêu giúp model học âm vị học.
"""

import os
import json

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from .tonal import (
    split_tone, merge_tone, apply_tone_to_syllable,
    tones_to_ids, TONE_NAMES, TONE_TO_ID,
)

# Các token đặc biệt — phải có mặt trong mọi từ điển.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class BigramTokenizer:
    """Tokenizer tách-thanh-điệu cho tiếng Việt."""

    def __init__(self, tokenizer: Tokenizer = None):
        # `tokenizer` là đối tượng BPE của thư viện HuggingFace.
        # Có thể None nếu ta sắp train tokenizer mới.
        self._tok = tokenizer

    # ------------------------------------------------------------------
    # Huấn luyện tokenizer
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text_files, vocab_size: int = 32000,
              min_frequency: int = 2) -> "BigramTokenizer":
        """
        Train một BPE tokenizer mới từ các file văn bản.

        QUAN TRỌNG: ta train BPE trên ÂM GỐC (đã bỏ thanh), không phải văn bản
        thô. Vì vậy file đầu vào được tách thanh trước, ghi ra file tạm, rồi
        mới đưa cho trainer.

        text_files   : danh sách đường dẫn file .txt (mỗi dòng một câu/đoạn).
        vocab_size   : kích thước từ điển mong muốn.
        min_frequency: tần suất tối thiểu để một cặp được gộp.
        """
        # Bước 1: tạo file tạm chứa âm gốc.
        base_files = []
        for path in text_files:
            base_path = path + ".base.tmp"
            with open(path, "r", encoding="utf-8") as fin, \
                 open(base_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    base, _ = split_tone(line.rstrip("\n"))
                    fout.write(base + "\n")
            base_files.append(base_path)

        # Bước 2: cấu hình BPE và train.
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        # Pre-tokenizer ByteLevel: làm việc ở mức byte -> xử lý được mọi ký tự.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(base_files, trainer)

        # Bước 3: dọn file tạm.
        for p in base_files:
            if os.path.exists(p):
                os.remove(p)

        return cls(tok)

    # ------------------------------------------------------------------
    # Lưu / nạp
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Lưu tokenizer ra một file JSON."""
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BigramTokenizer":
        """Nạp tokenizer từ file JSON."""
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    # ------------------------------------------------------------------
    # Mã hóa / giải mã
    # ------------------------------------------------------------------
    def encode(self, text: str, add_special: bool = False):
        """
        Mã hóa văn bản thành (token_ids, tone_ids).

        add_special: nếu True, thêm <bos> ở đầu và <eos> ở cuối.

        Cách căn thanh điệu: với mỗi token BPE, ta lấy lại đoạn ký tự gốc mà
        token đó biểu diễn (qua offsets), rồi tìm thanh điệu đầu tiên khác
        "<none>" trong đoạn đó.
        """
        # Tách thanh: lấy âm gốc + thanh điệu từng ký tự.
        base, char_tones = split_tone(text)

        # BPE mã hóa âm gốc. Bật offsets để biết token nào ứng với ký tự nào.
        enc = self._tok.encode(base)
        token_ids = list(enc.ids)
        offsets = enc.offsets  # list các (start, end) theo chỉ số ký tự của `base`.

        # Gán thanh điệu cho từng token.
        tone_ids = []
        for (start, end) in offsets:
            tone = "<none>"
            # Quét các ký tự trong khoảng [start, end) tìm thanh đầu tiên.
            for ci in range(start, min(end, len(char_tones))):
                if char_tones[ci] != "<none>":
                    tone = char_tones[ci]
                    break
            tone_ids.append(TONE_TO_ID[tone])

        # Thêm token đặc biệt nếu cần.
        if add_special:
            bos = self._tok.token_to_id("<bos>")
            eos = self._tok.token_to_id("<eos>")
            token_ids = [bos] + token_ids + [eos]
            # Token đặc biệt không mang thanh điệu -> id 0 (<none>).
            tone_ids = [0] + tone_ids + [0]

        return token_ids, tone_ids

    def decode(self, token_ids, tone_ids=None) -> str:
        """
        Giải mã (token_ids, tone_ids) trở lại văn bản có dấu.

        Nếu tone_ids = None thì chỉ trả về âm gốc (không có dấu thanh).
        """
        # Bỏ các token đặc biệt khỏi đầu ra văn bản.
        special_ids = {self._tok.token_to_id(t) for t in SPECIAL_TOKENS}
        special_ids.discard(None)

        # Lọc song song token_ids và tone_ids.
        if tone_ids is None:
            kept = [tid for tid in token_ids if tid not in special_ids]
            return self._tok.decode(kept)

        kept_tok, kept_tone = [], []
        for tid, tone in zip(token_ids, tone_ids):
            if tid not in special_ids:
                kept_tok.append(tid)
                kept_tone.append(tone)

        # Chiến lược decode đúng:
        # 1) Decode từng token thành chuỗi âm gốc, ghi lại độ dài để biết
        #    mỗi ký tự gốc thuộc token nào -> suy ra thanh điệu từng KÝ TỰ.
        # 2) Tách kết quả thành các âm tiết (theo khoảng trắng).
        # 3) Với mỗi âm tiết, lấy thanh điệu (thanh khác <none> đầu tiên trong
        #    âm tiết đó) rồi đặt dấu theo quy tắc chính tả tiếng Việt.
        base_chars = []   # từng ký tự của chuỗi âm gốc.
        char_tone = []    # thanh điệu tương ứng từng ký tự.
        for tid, tone in zip(kept_tok, kept_tone):
            piece = self._tok.decode([tid])
            tone_name = TONE_NAMES[tone] if 0 <= tone < len(TONE_NAMES) else "<none>"
            # Tìm chỉ số nguyên âm ĐẦU TIÊN trong piece để gắn thanh vào đó.
            # (piece có thể bắt đầu bằng khoảng trắng, nên không dùng index 0.)
            vowel_idx = -1
            for ci, ch in enumerate(piece):
                if ch.lower() in "aăâeêioôơuưy":
                    vowel_idx = ci
                    break
            for ci, ch in enumerate(piece):
                base_chars.append(ch)
                char_tone.append(tone_name if ci == vowel_idx else "<none>")

        base_text = "".join(base_chars)

        # Tách thành âm tiết, giữ nguyên khoảng trắng & dấu câu.
        result = []
        i = 0
        n = len(base_text)
        while i < n:
            ch = base_text[i]
            if ch.isalpha():
                # Gom một cụm chữ liền nhau = một âm tiết.
                j = i
                syl_chars = []
                syl_tone = "<none>"
                while j < n and base_text[j].isalpha():
                    syl_chars.append(base_text[j])
                    if char_tone[j] != "<none>":
                        syl_tone = char_tone[j]
                    j += 1
                result.append(apply_tone_to_syllable("".join(syl_chars), syl_tone))
                i = j
            else:
                # Khoảng trắng / dấu câu -> giữ nguyên.
                result.append(ch)
                i += 1

        return "".join(result)
