"""Classical (non-transformer) baselines that output per-character probabilities.

1. ``LexicalOverlap``: unsupervised. An answer sentence that resembles no context sentence
   (TF-IDF over character n-grams, which tolerates Turkish suffixes) is likely unsupported.
2. ``WordLogReg``: supervised logistic regression over hand-made word features
   (is the word / its stem / its number in the context, capitalization, sentence overlap ...).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

from .data import numbers
from .labels import char_mask
from .text import word_spans

_SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")


def tr_lower(text: str) -> str:
    """Turkish lowercasing for feature matching only (never for offsets)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for m in _SENT_RE.finditer(text):
        s, e = m.span()
        while s < e and text[s].isspace():
            s += 1
        if e - s >= 3:
            spans.append((s, e))
    return spans


class LexicalOverlap:
    """Score = 1 - max cosine similarity between an answer sentence and any context sentence."""

    def __init__(self, max_features: int = 200_000):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          max_features=max_features, sublinear_tf=True)

    def fit(self, prompts: Sequence[str], answers: Sequence[str]) -> "LexicalOverlap":
        self.vectorizer.fit([tr_lower(t) for t in list(prompts) + list(answers)])
        return self

    def sentence_scores(self, prompt: str, answer: str) -> tuple[list[tuple[int, int]], np.ndarray]:
        sents = sentence_spans(answer)
        ctx = [prompt[s:e] for s, e in sentence_spans(prompt)] or [prompt]
        if not sents:
            return [], np.zeros(0)
        a = self.vectorizer.transform([tr_lower(answer[s:e]) for s, e in sents])
        c = self.vectorizer.transform([tr_lower(t) for t in ctx])
        sim = linear_kernel(a, c).max(axis=1)
        return sents, 1.0 - np.asarray(sim).ravel()

    def predict_char_probs(self, prompt: str, answer: str) -> np.ndarray:
        probs = np.zeros(len(answer), dtype=np.float32)
        for (s, e), score in zip(*self.sentence_scores(prompt, answer)):
            probs[s:e] = score
        return probs


FEATURES = (
    "in_ctx", "stem_in_ctx", "is_number", "number_in_ctx", "capitalized", "sentence_initial",
    "length", "position", "sent_novelty", "sent_oov_share", "prev_stem_in_ctx", "next_stem_in_ctx",
)


def word_features(prompt: str, answer: str, overlap: LexicalOverlap) -> tuple[list[tuple[int, int]], np.ndarray]:
    """One feature row per answer word (see ``FEATURES``)."""
    spans = word_spans(answer)
    if not spans:
        return spans, np.zeros((0, len(FEATURES)), dtype=np.float32)
    ctx_words = {tr_lower(prompt[s:e]) for s, e in word_spans(prompt)}
    ctx_stems = {w[:5] for w in ctx_words}
    ctx_numbers = numbers(prompt)
    sents, novelty = overlap.sentence_scores(prompt, answer)

    def sentence_of(pos: int) -> int:
        for i, (s, e) in enumerate(sents):
            if s <= pos < e:
                return i
        return -1

    rows, stem_hits, sent_idx = [], [], []
    for s, e in spans:
        w = answer[s:e]
        lw = tr_lower(w)
        stem_hits.append(float(lw[:5] in ctx_stems))
        sent_idx.append(sentence_of(s))
        is_num = any(ch.isdigit() for ch in w)
        prev = answer[:s].rstrip()
        rows.append([
            float(lw in ctx_words), stem_hits[-1], float(is_num),
            float(is_num and bool(numbers(w) & ctx_numbers)),
            float(w[:1].isupper()), float(not prev or prev[-1] in ".!?\n"),
            min(len(w), 20) / 20, s / max(len(answer), 1), 0.0, 0.0, 0.0, 0.0,
        ])
    feats = np.array(rows, dtype=np.float32)
    stem_hits = np.array(stem_hits)
    sent_idx = np.array(sent_idx)
    for i in range(len(spans)):
        k = sent_idx[i]
        if k >= 0:
            feats[i, 8] = novelty[k]
            feats[i, 9] = 1.0 - stem_hits[sent_idx == k].mean()
        feats[i, 10] = stem_hits[i - 1] if i > 0 else 1.0
        feats[i, 11] = stem_hits[i + 1] if i + 1 < len(spans) else 1.0
    return spans, feats


class WordLogReg:
    def __init__(self, overlap: LexicalOverlap, C: float = 1.0):
        self.overlap = overlap
        self.model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)

    def fit(self, prompts, answers, gold_spans) -> "WordLogReg":
        xs, ys = [], []
        for prompt, answer, spans in zip(prompts, answers, gold_spans):
            wspans, feats = word_features(prompt, answer, self.overlap)
            if not wspans:
                continue
            mask = char_mask(len(answer), spans)
            xs.append(feats)
            ys.append([int(mask[s:e].any()) for s, e in wspans])
        self.model.fit(np.vstack(xs), np.concatenate(ys))
        return self

    def predict_char_probs(self, prompt: str, answer: str) -> np.ndarray:
        probs = np.zeros(len(answer), dtype=np.float32)
        wspans, feats = word_features(prompt, answer, self.overlap)
        if wspans:
            for (s, e), p in zip(wspans, self.model.predict_proba(feats)[:, 1]):
                probs[s:e] = p
        return probs

    def coefficients(self) -> dict:
        return dict(zip(FEATURES, self.model.coef_[0].round(3).tolist()))
