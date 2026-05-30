"""
vs_bpe.py
=========
VietnameseSyllableAwareTokenizer — Tokenizer BPE đơn luồng (Single-Stream)
nhận biết âm tiết tiếng Việt, bảo đảm không cắt rời âm tiết và hoàn toàn tương thích
với vLLM, Hugging Face và các hệ sinh thái Transformer chuẩn.
"""

import os
import re
import unicodedata
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Regex khớp chính xác các từ/âm tiết tiếng Việt (chuẩn NFC/NFD) và các cụm ký tự khác
VIETNAMESE_SYLLABLE_PATTERN = re.compile(
    r"[A-Za-zđĐĂăÂâÊêÔôƠơƯưÀàẢảÃãÁáẠạẰằẲẳẴẵẮắẶặẦầẨẩẪẫẤấẬậỀềỂểỄễẾếỆệỈỉĨĩÍíỊịÒòỎỏÕõÓóỌọỒồỔổỖỗỐốỘộỜờỞởỠỡỚớỢợÙùỦủŨũÚúỤụỪừỬửỮữỨứỰựỲỳỶỷỸỹÝýỴỵ]+"
    r"|[^A-Za-zđĐĂăÂâÊêÔôƠơƯưÀàẢảÃãÁáẠạẰằẲẳẴẵẮắẶặẦầẨẩẪẫẤấẬậỀềỂểỄễẾếỆệỈỉĨĩÍíỊịÒòỎỏÕõÓóỌọỒồỔổỖỗỐốỘộỜờỞởỠỡỚớỢợÙùỦủŨũÚúỤụỪừỬửỮữỨứỰựỲỳỶỷỸỹÝýỴỵ\s]+"
    r"|\s+"
)

class VietnameseSyllableAwareTokenizer:
    """Tokenizer BPE nhận biết âm tiết tiếng Việt, định dạng Single-Stream."""

    def __init__(self, tokenizer: Tokenizer = None):
        self._tok = tokenizer

    @classmethod
    def train(cls, text_files, vocab_size: int = 32000,
              min_frequency: int = 2) -> "VietnameseSyllableAwareTokenizer":
        """
        Huấn luyện BPE tokenizer nhận biết âm tiết từ các file văn bản.
        """
        # Bước 1: Tiền phân đoạn âm tiết và ghi ra file tạm để BPE học chuẩn
        segmented_files = []
        for path in text_files:
            seg_path = path + ".vs_seg.tmp"
            with open(path, "r", encoding="utf-8") as fin, \
                 open(seg_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    normalized = unicodedata.normalize("NFC", line.rstrip("\n"))
                    # Tìm tất cả các âm tiết/phần tử
                    tokens = VIETNAMESE_SYLLABLE_PATTERN.findall(normalized)
                    # Ghi lại dưới dạng các phần tử phân tách rõ ràng để BPE không gộp sai ranh giới
                    fout.write(" ".join(tokens) + "\n")
            segmented_files.append(seg_path)

        # Bước 2: Khởi tạo và huấn luyện BPE
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        # Sử dụng pre-tokenizer ByteLevel nhưng kèm split để giữ nguyên ranh giới âm tiết
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=VIETNAMESE_SYLLABLE_PATTERN.pattern,
                behavior="removed"  # split theo pattern
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False)
        ])
        tok.decoder = decoders.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(segmented_files, trainer)

        # Bước 3: Dọn dẹp file tạm
        for p in segmented_files:
            if os.path.exists(p):
                os.remove(p)

        return cls(tok)

    def save(self, path: str):
        """Lưu tokenizer dưới dạng file JSON của Hugging Face."""
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "VietnameseSyllableAwareTokenizer":
        """Nạp tokenizer từ file JSON."""
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    def encode(self, text: str, add_special: bool = False):
        """
        Mã hóa văn bản thành (token_ids, None).
        Hoàn toàn tương thích đơn luồng (Single-Stream).
        """
        normalized = unicodedata.normalize("NFC", text)
        enc = self._tok.encode(normalized)
        token_ids = list(enc.ids)

        if add_special:
            bos = self._tok.token_to_id("<bos>")
            eos = self._tok.token_to_id("<eos>")
            token_ids = [bos] + token_ids + [eos]

        return token_ids, None

    def decode(self, token_ids, tone_ids=None) -> str:
        """
        Giải mã token_ids thành văn bản NFC.
        Chấp nhận tham số tone_ids=None để tương thích ngược API.
        """
        special_ids = {self._tok.token_to_id(t) for t in SPECIAL_TOKENS if self._tok.token_to_id(t) is not None}
        kept_tok = [tid for tid in token_ids if tid not in special_ids]
        decoded = self._tok.decode(kept_tok)
        return unicodedata.normalize("NFC", decoded)
