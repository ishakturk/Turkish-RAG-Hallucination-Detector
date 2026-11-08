"""Conversion between character spans and token labels, and back to character probabilities.

All offsets refer to the original answer string. Text is never lowercased here: Python's
``"İ".lower()`` returns two characters, which would shift every later offset.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

IGNORE_INDEX = -100

Span = dict  # {"start": int, "end": int, ...}


def char_mask(text_len: int, spans: Iterable[Span]) -> np.ndarray:
    """Boolean mask over the answer characters that fall inside any span."""
    mask = np.zeros(text_len, dtype=bool)
    for span in spans:
        start = max(0, int(span["start"]))
        end = min(text_len, int(span["end"]))
        if end > start:
            mask[start:end] = True
    return mask


def label_tokens(
    offsets: Sequence[tuple[int, int]],
    is_answer: Sequence[bool],
    spans: Iterable[Span],
    answer_len: int,
) -> list[int]:
    """Token labels: -100 outside the answer, 1 if the token overlaps a span, else 0.

    ``offsets`` must be relative to the answer string for answer tokens (Hugging Face fast
    tokenizers do this for the second sequence of a pair).
    """
    mask = char_mask(answer_len, spans)
    labels = []
    for (start, end), in_answer in zip(offsets, is_answer):
        if not in_answer or end <= start:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(int(mask[start:end].any()))
    return labels


def encode_example(
    tokenizer,
    prompt: str,
    answer: str,
    spans: Iterable[Span] = (),
    max_length: int = 8192,
) -> dict:
    """Tokenize ``[CLS] prompt [SEP] answer [SEP]`` and build labels.

    Only the prompt is truncated, never the answer. Returns input ids, attention mask, labels,
    the answer-relative offsets (needed to map predictions back to characters) and whether the
    prompt was cut.
    """
    enc = tokenizer(
        prompt,
        answer,
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    sequence_ids = enc.sequence_ids()
    is_answer = [sid == 1 for sid in sequence_ids]
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    full_len = len(tokenizer(prompt, answer)["input_ids"])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": label_tokens(offsets, is_answer, spans, len(answer)),
        "answer_offsets": [o if a else None for o, a in zip(offsets, is_answer)],
        "truncated": full_len > max_length,
    }


def token_probs_to_char_probs(
    answer_len: int,
    answer_offsets: Sequence[tuple[int, int] | None],
    token_probs: Sequence[float],
) -> np.ndarray:
    """Spread each answer token's hallucination probability over the characters it covers.

    Characters covered by no token (usually spaces) get 0. When tokens overlap, the
    higher probability wins.
    """
    probs = np.zeros(answer_len, dtype=np.float32)
    for offset, p in zip(answer_offsets, token_probs):
        if offset is None:
            continue
        start, end = offset
        if end > start:
            probs[start:end] = np.maximum(probs[start:end], p)
    return probs
