"""Scoring on the native Turkish test set (notebook 10, analysis plan §6 and §10).

The native set is scored once, with each system's frozen dev thresholds (results/07_thresholds.json).
Primary: example F1 and example AUROC; secondary: span F1 (gold spans are LLM-proposed).
Bootstrap resamples contexts: all questions about one context move together.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import _auroc_items

# Existing detectors: Hugging Face id and whether they need trust_remote_code (as in notebook 02).
EXISTING_MODELS = {
    "newmind_modernbert_tr": ("newmindai/ModernBERT-tr-uncased-stsb-HD", False),
    "newmind_turkembed4sts": ("newmindai/TurkEmbed4STS-HD", True),
    "newmind_eurobert_tr": ("newmindai/lettucedect-210m-eurobert-tr-v1", True),
    "newmind_bge_m3": ("newmindai/bge-m3-HD", False),
    "newmind_ettin_150m": ("newmindai/ettin-encoder-150M-TR-HD", False),
    "newmind_ettin_32m": ("newmindai/ettin-encoder-32M-TR-HD", False),
    "krlabs_v2_mmbert_zeroshot": ("KRLabsOrg/lettucedect-v2-mmbert-base", False),
}


def context_key(passages) -> str:
    return "\n\n".join(str(p) for p in passages)


def native_frame(test: pd.DataFrame, reference: pd.DataFrame, spans: dict, sheet_answers: dict) -> pd.DataFrame:
    """The native test set with gold spans. ``test`` is native_test.parquet; ``reference`` has id and
    label; ``spans`` maps id -> [{start, end, ...}]; ``sheet_answers`` maps id -> the answer text that
    was labeled (from the committed blank sheet). Any mismatch raises."""
    frame = test.set_index("id", drop=False).copy()
    ref = dict(zip(reference["id"], reference["label"].astype(int)))
    if set(frame.index) != set(ref):
        raise ValueError("native_test.parquet and the reference labels cover different ids")
    changed = [i for i in frame.index if frame.loc[i, "answer"] != sheet_answers[i]]
    if changed:
        raise ValueError(f"answers differ from the labeled sheet: {changed[:5]}")
    for i, label in ref.items():
        if bool(spans.get(i)) != bool(label):
            raise ValueError(f"{i}: reference label {label} but {len(spans.get(i, []))} spans")
        for s in spans.get(i, []):
            if "text" in s and frame.loc[i, "answer"][s["start"]:s["end"]] != s["text"]:
                raise ValueError(f"{i}: span offsets do not match the answer text")
    frame["labels"] = [[{"start": int(s["start"]), "end": int(s["end"]), "label": "native"}
                        for s in spans.get(i, [])] for i in frame.index]
    frame["reference_label"] = [ref[i] for i in frame.index]
    frame["task_type"] = frame["domain"]                 # per-"task" results are per domain here
    frame["source_id"] = [context_key(p) for p in frame["passages"]]
    return frame


def example_auroc_ci(runs: Sequence[Sequence[dict]], n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Seed-averaged example AUROC (max character probability) with a percentile CI, resampling the
    records' ``group`` once per replicate for all runs. Replicates with one class only are skipped."""
    items = [[_auroc_items(r) for r in run] for run in runs]
    groups: dict = {}
    for k, r in enumerate(runs[0]):
        groups.setdefault(r["group"], []).append(k)
    blocks = list(groups.values())
    y = np.array([it[0] for it in items[0]])
    scores = np.array([[it[1] for it in run] for run in items])

    def avg(idx):
        if len(set(y[idx])) < 2:
            return None
        return float(np.mean([roc_auc_score(y[idx], s[idx]) for s in scores]))

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[b] for b in rng.integers(0, len(blocks), len(blocks))])
        v = avg(idx)
        if v is not None:
            stats.append(v)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": avg(np.arange(len(y))), "low": float(lo), "high": float(hi),
            "n_groups": len(blocks), "replicates_used": len(stats)}
