"""
Package `tokenizer` — xử lý văn bản tiếng Việt cho Bigram.

  - tonal.py : tách / ghép thanh điệu tiếng Việt.
  - bpe.py   : BigramTokenizer (BPE trên âm gốc + căn thanh điệu).
"""

from .bpe import BigramTokenizer, SPECIAL_TOKENS
from .bmssp import BMSSPTokenizer
from .vs_bpe import VietnameseSyllableAwareTokenizer
from .tonal import (
    split_tone, merge_tone, apply_tone_to_syllable,
    tones_to_ids, ids_to_tones, TONE_NAMES, TONE_TO_ID,
)

__all__ = [
    "BigramTokenizer",
    "BMSSPTokenizer",
    "VietnameseSyllableAwareTokenizer",
    "SPECIAL_TOKENS",
    "split_tone",
    "merge_tone",
    "apply_tone_to_syllable",
    "tones_to_ids",
    "ids_to_tones",
    "TONE_NAMES",
    "TONE_TO_ID",
]
