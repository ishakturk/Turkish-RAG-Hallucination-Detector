"""Turn character probabilities into predicted spans aligned to Turkish word boundaries."""

from __future__ import annotations

import numpy as np

from .text import word_spans


def word_probs(answer: str, char_probs: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Word boundaries and each word's score (max over its characters)."""
    spans = word_spans(answer)
    scores = np.array(
        [float(char_probs[s:e].max()) if e > s else 0.0 for s, e in spans], dtype=np.float32
    )
    return spans, scores


def char_probs_to_spans(
    answer: str,
    char_probs: np.ndarray,
    threshold: float = 0.5,
    bridge_max_chars: int = 3,
    min_chars: int = 2,
) -> list[dict]:
    """Flag words whose score reaches ``threshold`` and merge them into spans.

    Two flagged words are merged when only punctuation/whitespace separates them, or a single
    short unflagged word (at most ``bridge_max_chars`` characters, e.g. "ve", "bir") sits between
    them. Hallucinated tokens come in contiguous runs, so this repairs fragmented predictions.
    """
    spans, scores = word_probs(answer, char_probs)
    flagged = scores >= threshold
    result: list[list[int]] = []
    i = 0
    while i < len(spans):
        if not flagged[i]:
            i += 1
            continue
        start, end = spans[i]
        j = i + 1
        while j < len(spans):
            if flagged[j]:
                end = spans[j][1]
                j += 1
            elif (
                j + 1 < len(spans)
                and flagged[j + 1]
                and spans[j][1] - spans[j][0] <= bridge_max_chars
            ):
                end = spans[j + 1][1]
                j += 2
            else:
                break
        result.append([start, end])
        i = j
    return [
        {"start": s, "end": e, "text": answer[s:e], "confidence": float(char_probs[s:e].max())}
        for s, e in result
        if e - s >= min_chars
    ]
