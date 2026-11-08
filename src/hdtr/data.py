"""RAGTruth-TR loading, alignment to the English RAGTruth, data audit and grouped splits."""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .text import digits, ends_like_sentence, word_spans, words

RAGTRUTH_TR = "newmindai/RAGTruth-TR"
RAGTRUTH_EN_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
TASKS = ("QA", "Data2txt", "Summary")


def _as_spans(labels) -> list[dict]:
    """Normalize label lists coming from pandas/arrow (arrays of dicts) to plain dicts."""
    if labels is None:
        return []
    return [
        {"start": int(l["start"]), "end": int(l["end"]), "label": str(l.get("label", ""))}
        for l in list(labels)
    ]


def load_ragtruth_tr() -> pd.DataFrame:
    """The 17,790 Turkish responses; ``split`` is a column (train 15,090 / test 2,700)."""
    from datasets import load_dataset

    df = load_dataset(RAGTRUTH_TR, split="train").to_pandas()
    df["labels"] = df["labels"].map(_as_spans)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def _read_jsonl(url: str, cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ragtruth_en(cache_dir: str | Path = "data/cache") -> pd.DataFrame:
    """Original English RAGTruth responses joined with their sources (in response-file order)."""
    cache_dir = Path(cache_dir)
    responses = _read_jsonl(RAGTRUTH_EN_URL + "response.jsonl", cache_dir)
    sources = {s["source_id"]: s for s in _read_jsonl(RAGTRUTH_EN_URL + "source_info.jsonl", cache_dir)}
    rows = []
    for r in responses:
        src = sources[r["source_id"]]
        rows.append({
            "response_id": r["id"],
            "source_id": r["source_id"],
            "model": r.get("model"),
            "quality": r.get("quality"),
            "split": r["split"],
            "task_type": src["task_type"],
            "prompt": src.get("prompt", ""),
            "answer": r["response"],
            "labels": [
                {"start": int(l["start"]), "end": int(l["end"]), "label": l.get("label_type", "")}
                for l in r.get("labels", [])
            ],
        })
    return pd.DataFrame(rows)


# --- Alignment -------------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    """Numbers in ``text`` with separators removed, so "1.250" and "1,250" match."""
    return {digits(m) for m in _NUMBER_RE.findall(text)}


def _digits_agree(a: str, b: str) -> bool:
    da, db = digits(a), digits(b)
    if max(len(da), len(db)) < 3:
        return True
    return SequenceMatcher(None, da, db).ratio() >= 0.8


def alignment_signals(tr: pd.DataFrame, en: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row agreement signals between two equally long, already paired frames."""
    if len(tr) != len(en):
        raise ValueError(f"length mismatch: {len(tr)} vs {len(en)}")
    tr = tr.reset_index(drop=True)
    en = en.reset_index(drop=True)
    len_ratio = tr["answer"].str.len() / en["answer"].str.len().clip(lower=1)
    out = pd.DataFrame({
        "row_id": tr["row_id"],
        "source_id": en["source_id"],
        "response_id": en["response_id"],
        "same_split": tr["split"].values == en["split"].values,
        "same_task": tr["task_type"].values == en["task_type"].values,
        "n_spans_tr": tr["labels"].map(len),
        "n_spans_en": en["labels"].map(len),
        "len_ratio": len_ratio,
        "digits_agree": [_digits_agree(a, b) for a, b in zip(tr["answer"], en["answer"])],
    })
    out["same_n_spans"] = out["n_spans_tr"] == out["n_spans_en"]
    # Digits are reported but not required: translation legitimately rewrites numbers
    # ("5 pm" -> "17:00", added Celsius conversions), so they are an audit signal instead.
    out["pass"] = (
        out["same_split"] & out["same_task"]
        & out["len_ratio"].between(0.6, 1.8)
        & ((out["n_spans_tr"] > 0) == (out["n_spans_en"] > 0))
    )
    return out


def align_by_order(tr: pd.DataFrame, en: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Pair rows by position, optionally within groups (e.g. ``("split", "task_type")``)."""
    if not by:
        return alignment_signals(tr, en)
    parts = []
    en_groups = dict(list(en.groupby(list(by), sort=False)))
    for key, tr_part in tr.groupby(list(by), sort=False):
        en_part = en_groups.get(key, en.iloc[:0])
        if len(en_part) != len(tr_part):
            raise ValueError(f"group {key}: {len(tr_part)} TR rows vs {len(en_part)} EN rows")
        parts.append(alignment_signals(tr_part, en_part))
    return pd.concat(parts, ignore_index=True)


# --- Audit -----------------------------------------------------------------------------------

# Frequent English function words that are not also common Turkish words.
EN_STOPWORDS = {
    "the", "and", "of", "to", "is", "in", "that", "it", "for", "with", "was", "are", "this",
    "by", "at", "from", "or", "which", "have", "has", "not", "but", "they", "their", "were",
    "been", "will", "would", "can", "there", "also", "its", "more", "than", "these", "such",
    "about", "into", "who", "what", "when", "where", "you", "your", "our", "we",
}

HARD_FLAGS = ("span_out_of_bounds", "span_empty", "span_count_mismatch", "repetition", "truncated")
SOFT_FLAGS = ("len_ratio_out", "english_residue", "number_added")


def _english_share(text: str) -> float:
    ws = [w.lower() for w in words(text)]
    return sum(w in EN_STOPWORDS for w in ws) / len(ws) if ws else 0.0


def _max_ngram_count(text: str, n: int = 4) -> int:
    ws = [w.lower() for w in words(text)]
    grams = Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))
    return max(grams.values(), default=0)


def _has_repetition(text: str, en_text: str | None = None, times: int = 6) -> bool:
    """Degenerate looping: one 4-gram repeated ``times``+ times and (if known) over twice as
    often as the English original's most repeated 4-gram. Business names repeated in both
    languages are therefore not flagged."""
    count = _max_ngram_count(text)
    if count < times:
        return False
    return en_text is None or count > 2 * _max_ngram_count(en_text)


def _mid_word(text: str, pos: int) -> bool:
    return 0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum()


def added_numbers(answer: str, spans: list[dict], en_answer: str, prompt: str = "") -> set[str]:
    """Numbers the translation introduced: absent from the English answer and the context,
    and outside any labeled span. These are unlabeled unsupported facts (label noise)."""
    in_spans = set().union(*(numbers(answer[s["start"]:s["end"]]) for s in spans)) if spans else set()
    return numbers(answer) - numbers(en_answer) - numbers(prompt) - in_spans


def audit_row(answer: str, spans: list[dict], en_answer: str | None = None,
              en_spans: list[dict] | None = None, prompt: str = "") -> dict:
    """Quality flags for one translated response (see docs/technical-roadmap.md, 2.5)."""
    n = len(answer)
    flags = {
        "span_out_of_bounds": any(s["start"] < 0 or s["end"] > n or s["end"] <= s["start"] for s in spans),
        "span_empty": any(not answer[s["start"]:s["end"]].strip() for s in spans if 0 <= s["start"] < s["end"] <= n),
        "span_midword": any(_mid_word(answer, s["start"]) or _mid_word(answer, s["end"]) for s in spans),
        "english_residue": _english_share(answer) > 0.15,
        "repetition": _has_repetition(answer, en_answer),
        "span_count_mismatch": False,
        "len_ratio_out": False,
        "truncated": False,
        "number_added": False,
    }
    if en_answer is not None:
        ratio = len(answer) / max(len(en_answer), 1)
        flags["len_ratio_out"] = not 0.5 <= ratio <= 2.0
        flags["truncated"] = not ends_like_sentence(answer) and ends_like_sentence(en_answer)
        flags["number_added"] = bool(added_numbers(answer, spans, en_answer, prompt))
    if en_spans is not None:
        flags["span_count_mismatch"] = len(spans) != len(en_spans)
    return flags


def drop_decision(flags: dict) -> bool:
    """Hard flags always drop; soft flags drop only together with another flag."""
    if any(flags[f] for f in HARD_FLAGS):
        return True
    soft = sum(flags[f] for f in SOFT_FLAGS)
    return soft >= 1 and (soft + flags["span_midword"]) >= 2


# --- Splits ----------------------------------------------------------------------------------

def make_splits(df: pd.DataFrame, group_col: str = "source_id", dev_sources_per_task: int = 30,
                val_frac: float = 0.05, seed: int = 482) -> dict[str, list[int]]:
    """Grouped splits by source: train/validation from the train split, dev/test from test.

    dev = ``dev_sources_per_task`` sources per task (90 sources, 540 responses by default);
    test = the remaining test sources. All row ids of one source stay in one split.
    """
    rng = np.random.default_rng(seed)
    out = {"train": [], "validation": [], "dev": [], "test": []}
    for (split, task), part in df.groupby(["split", "task_type"]):
        groups = np.array(sorted(part[group_col].unique()))
        rng.shuffle(groups)
        if split == "test":
            held = set(groups[:dev_sources_per_task])
            names = ("dev", "test")
        else:
            held = set(groups[: max(1, round(len(groups) * val_frac))])
            names = ("validation", "train")
        in_held = part[group_col].isin(held)
        out[names[0]] += part.loc[in_held, "row_id"].tolist()
        out[names[1]] += part.loc[~in_held, "row_id"].tolist()
    return {k: sorted(int(i) for i in v) for k, v in out.items()}
