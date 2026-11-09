"""Conversion between character spans and token labels, and back to character probabilities.

All offsets refer to the original answer string. Text is never lowercased here: Python's
``"İ".lower()`` returns two characters, which would shift every later offset.
"""

from __future__ import annotations

import re
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


# A chunk is one whitespace-delimited piece plus its trailing whitespace (TabiBERT's BPE
# attaches spaces at the end of tokens, e.g. "ular ").
_CHUNK_RE = re.compile(r"\S+\s*|\s+")


def _chunked_ids(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize each whitespace chunk separately so no token can span two words."""
    chunks = [(m.start(), m.group()) for m in _CHUNK_RE.finditer(text)]
    if not chunks:
        return [], []
    enc = tokenizer([c for _, c in chunks], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = [], []
    for (start, _), chunk_ids, chunk_offsets in zip(chunks, enc["input_ids"], enc["offset_mapping"]):
        ids += chunk_ids
        offsets += [(s + start, e + start) for s, e in chunk_offsets]
    return ids, offsets


def _pair_layout(tokenizer) -> list:
    """Special-token layout of a pair, e.g. [cls_id, 0, sep_id, 1, sep_id] (0/1 = segments)."""
    enc = tokenizer("a", "b")
    layout = []
    for token_id, seq in zip(enc["input_ids"], enc.sequence_ids()):
        item = ("seg", seq) if seq is not None else ("id", token_id)
        if not layout or item[0] == "id" or layout[-1] != item:
            layout.append(item)
    return layout


def _encode_split_words(tokenizer, prompt: str, answer: str, spans, max_length: int) -> dict:
    p_ids, _ = _chunked_ids(tokenizer, prompt)
    a_ids, a_offsets = _chunked_ids(tokenizer, answer)
    layout = _pair_layout(tokenizer)
    n_special = sum(kind == "id" for kind, _ in layout)
    full_len = n_special + len(p_ids) + len(a_ids)
    keep = max(0, max_length - n_special - len(a_ids))
    p_ids = p_ids[:keep]
    input_ids, offsets, is_answer = [], [], []
    for kind, value in layout:
        if kind == "id":
            input_ids.append(value); offsets.append((0, 0)); is_answer.append(False)
        elif value == 0:
            input_ids += p_ids; offsets += [(0, 0)] * len(p_ids); is_answer += [False] * len(p_ids)
        else:
            input_ids += a_ids; offsets += a_offsets; is_answer += [True] * len(a_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": label_tokens(offsets, is_answer, spans, len(answer)),
        "answer_offsets": [o if a else None for o, a in zip(offsets, is_answer)],
        "truncated": full_len > max_length,
    }


def encode_example(
    tokenizer,
    prompt: str,
    answer: str,
    spans: Iterable[Span] = (),
    max_length: int = 8192,
    split_words: bool = False,
) -> dict:
    """Tokenize ``[CLS] prompt [SEP] answer [SEP]`` and build labels.

    Only the prompt is truncated, never the answer. Returns input ids, attention mask, labels,
    the answer-relative offsets (needed to map predictions back to characters) and whether the
    prompt was cut. ``split_words=True`` tokenizes whitespace chunks separately, so tokenizers
    that merge across words (TabiBERT: "a kadar") get word-aligned tokens.
    """
    spans = list(spans)
    if split_words:
        return _encode_split_words(tokenizer, prompt, answer, spans, max_length)
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
