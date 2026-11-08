"""Turkish-aware word segmentation shared by labels, post-processing and metrics.

Words are the common unit across models with different tokenizers, so every model is
scored on the same word boundaries. An apostrophe suffix stays with its stem
("Ocak'a", "TL'dir") and numbers keep their separators ("15.000", "3,5").
"""

from __future__ import annotations

import re

APOSTROPHES = "'’"

# Numbers first so "15.000'e" is one word; then letter/digit runs joined by apostrophes.
_WORD_RE = re.compile(
    rf"\d+(?:[.,]\d+)+(?:[{APOSTROPHES}]\w+)?"
    rf"|\w+(?:[{APOSTROPHES}]\w+)*"
)

_SENTENCE_END_RE = re.compile(r"[.!?…:)\"”»]\s*$")


def word_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of every word in ``text``."""
    return [m.span() for m in _WORD_RE.finditer(text)]


def words(text: str) -> list[str]:
    return [text[s:e] for s, e in word_spans(text)]


def ends_like_sentence(text: str) -> bool:
    """True when ``text`` ends with sentence-final punctuation (used to spot truncation)."""
    return bool(_SENTENCE_END_RE.search(text.strip())) if text.strip() else False


def digits(text: str) -> str:
    """All digits of ``text`` in order, ignoring separators (1.000 and 1,000 both give 1000)."""
    return "".join(ch for ch in text if ch.isdigit())
