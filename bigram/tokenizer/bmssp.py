"""
bmssp.py
========
BMSSPTokenizer — tokenizer sử dụng SentencePiece Unigram với Byte-fallback
và Unicode NFC normalization, ép phân đoạn tại ranh giới âm tiết tiếng Việt.
"""

import os
import sentencepiece as spm
import tempfile
import unicodedata

class BMSSPTokenizer:
    """Tokenizer sử dụng SentencePiece Unigram với Byte-fallback và Unicode NFC normalization."""

    def __init__(self, sp_processor=None):
        self.sp = sp_processor if sp_processor is not None else spm.SentencePieceProcessor()

    @classmethod
    def train(cls, text_files, vocab_size: int = 8000) -> "BMSSPTokenizer":
        # 1. Chuẩn hóa NFC và ghi ra file tạm
        temp_corpus = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.txt')
        try:
            for path in text_files:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        normalized = unicodedata.normalize("NFC", line)
                        temp_corpus.write(normalized)
            temp_corpus.close()

            # 2. Huấn luyện SentencePiece
            model_prefix = tempfile.mktemp()
            spm.SentencePieceTrainer.train(
                input=temp_corpus.name,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="unigram",
                byte_fallback=True,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<bos>",
                eos_piece="<eos>",
                split_by_whitespace=True,
                allow_whitespace_only_pieces=True,
                treat_whitespace_as_suffix=False,
            )

            # 3. Đọc model đã train và xóa file tạm
            model_path = model_prefix + ".model"
            sp = spm.SentencePieceProcessor()
            with open(model_path, "rb") as f:
                proto = f.read()
            sp.load_from_serialized_proto(proto)
            
            # Xóa các file tạm của model
            if os.path.exists(model_path):
                os.remove(model_path)
            vocab_path = model_prefix + ".vocab"
            if os.path.exists(vocab_path):
                os.remove(vocab_path)
        finally:
            if os.path.exists(temp_corpus.name):
                os.remove(temp_corpus.name)

        return cls(sp)

    def save(self, path: str):
        """Lưu model dưới dạng file nhị phân SentencePiece (.model)."""
        proto = self.sp.serialized_model_proto()
        with open(path, "wb") as f:
            f.write(proto)

    @classmethod
    def load(cls, path: str) -> "BMSSPTokenizer":
        """Nạp model từ file nhị phân SentencePiece."""
        sp = spm.SentencePieceProcessor()
        with open(path, "rb") as f:
            proto = f.read()
        sp.load_from_serialized_proto(proto)
        return cls(sp)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def encode(self, text: str, add_special: bool = False):
        """Mã hóa văn bản thành (token_ids, tone_ids=None)."""
        normalized = unicodedata.normalize("NFC", text)
        token_ids = self.sp.encode(normalized)
        if add_special:
            bos = self.sp.bos_id()
            eos = self.sp.eos_id()
            token_ids = [bos] + token_ids + [eos]
        return token_ids, None

    def decode(self, token_ids, tone_ids=None) -> str:
        """Giải mã token_ids thành văn bản chuẩn NFC."""
        special_ids = {0, 1, 2, 3}
        filtered_ids = [tid for tid in token_ids if tid not in special_ids]
        decoded = self.sp.decode(filtered_ids)
        return unicodedata.normalize("NFC", decoded)
