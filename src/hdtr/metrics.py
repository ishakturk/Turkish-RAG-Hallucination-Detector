"""Example-, word- and span-level metrics, plus source-grouped bootstrap and permutation tests.

Every system is scored from the same inputs per example: the answer text, gold spans,
predicted spans and (optionally) per-character probabilities. Scoring at word and character
level keeps models with different tokenizers comparable.

A record is a dict with keys:
    answer (str), gold_spans (list), pred_spans (list),
    char_probs (np.ndarray, optional), task_type (str, optional), group (hashable, optional)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .labels import char_mask
from .text import word_spans

COUNT_KEYS = (
    "ex_tp", "ex_fp", "ex_fn", "ex_tn",
    "w_tp", "w_fp", "w_fn", "w_tn",
    "c_overlap", "c_pred", "c_gold",
)


def record_counts(record: dict) -> dict:
    """Additive counts for one example; metrics are computed from sums of these."""
    answer = record["answer"]
    gold = char_mask(len(answer), record["gold_spans"])
    pred = char_mask(len(answer), record["pred_spans"])

    gold_ex = bool(gold.any())
    pred_ex = bool(pred.any())

    w_tp = w_fp = w_fn = w_tn = 0
    for s, e in word_spans(answer):
        g = bool(gold[s:e].any())
        p = bool(pred[s:e].any())
        w_tp += g and p
        w_fp += (not g) and p
        w_fn += g and not p
        w_tn += (not g) and not p

    return {
        "ex_tp": int(gold_ex and pred_ex),
        "ex_fp": int(pred_ex and not gold_ex),
        "ex_fn": int(gold_ex and not pred_ex),
        "ex_tn": int(not gold_ex and not pred_ex),
        "w_tp": w_tp, "w_fp": w_fp, "w_fn": w_fn, "w_tn": w_tn,
        "c_overlap": int((gold & pred).sum()),
        "c_pred": int(pred.sum()),
        "c_gold": int(gold.sum()),
    }


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def metrics_from_counts(c: dict) -> dict:
    """Hallucinated-class P/R/F1 at each level, plus macro F1 for example and word level."""
    out = {}
    for level, pre in (("example", "ex"), ("word", "w")):
        p, r, f = _prf(c[f"{pre}_tp"], c[f"{pre}_fp"], c[f"{pre}_fn"])
        # Supported class: swap roles of positives and negatives.
        _, _, f_neg = _prf(c[f"{pre}_tn"], c[f"{pre}_fn"], c[f"{pre}_fp"])
        out[level] = {"precision": p, "recall": r, "f1": f, "macro_f1": (f + f_neg) / 2}
    sp = c["c_overlap"] / c["c_pred"] if c["c_pred"] else 0.0
    sr = c["c_overlap"] / c["c_gold"] if c["c_gold"] else 0.0
    sf = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
    out["span"] = {"precision": sp, "recall": sr, "f1": sf}
    out["example"]["confusion"] = [[c["ex_tn"], c["ex_fp"]], [c["ex_fn"], c["ex_tp"]]]
    return out


def sum_counts(counts: Iterable[dict]) -> dict:
    total = dict.fromkeys(COUNT_KEYS, 0)
    for c in counts:
        for k in COUNT_KEYS:
            total[k] += c[k]
    return total


def auroc(records: Sequence[dict]) -> dict:
    """AUROC from probabilities: example level (max char prob) and word level (max per word)."""
    ex_y, ex_s, w_y, w_s = [], [], [], []
    for r in records:
        probs = r.get("char_probs")
        if probs is None:
            return {}
        answer = r["answer"]
        gold = char_mask(len(answer), r["gold_spans"])
        ex_y.append(int(gold.any()))
        ex_s.append(float(probs.max()) if len(probs) else 0.0)
        for s, e in word_spans(answer):
            w_y.append(int(gold[s:e].any()))
            w_s.append(float(probs[s:e].max()))
    out = {}
    if len(set(ex_y)) == 2:
        out["example_auroc"] = float(roc_auc_score(ex_y, ex_s))
    if len(set(w_y)) == 2:
        out["word_auroc"] = float(roc_auc_score(w_y, w_s))
    return out


def evaluate(records: Sequence[dict], by_task: bool = True) -> dict:
    """Metrics overall and per ``task_type``."""
    counts = [record_counts(r) for r in records]
    result = {"overall": {**metrics_from_counts(sum_counts(counts)), **auroc(records)}}
    result["overall"]["n"] = len(records)
    if by_task:
        tasks = defaultdict(list)
        for r, c in zip(records, counts):
            tasks[r.get("task_type", "all")].append((r, c))
        for task, items in sorted(tasks.items()):
            recs = [r for r, _ in items]
            result[task] = {
                **metrics_from_counts(sum_counts(c for _, c in items)),
                **auroc(recs),
                "n": len(recs),
            }
    return result


def _grouped_counts(records: Sequence[dict]) -> tuple[list, np.ndarray]:
    """Per-group count vectors (groups x COUNT_KEYS), in first-seen group order."""
    groups: dict = {}
    for i, r in enumerate(records):
        vec = np.array([record_counts(r)[k] for k in COUNT_KEYS], dtype=np.int64)
        key = r.get("group", i)
        groups[key] = groups.get(key, 0) + vec
    keys = list(groups)
    return keys, np.stack([groups[k] for k in keys])


def _as_counts(vec: np.ndarray) -> dict:
    return dict(zip(COUNT_KEYS, vec.tolist()))


MetricFn = Callable[[dict], float]


def metric(level: str, name: str = "f1") -> MetricFn:
    return lambda counts: metrics_from_counts(counts)[level][name]


def bootstrap_ci(
    records: Sequence[dict],
    fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI, resampling source groups (responses of one source move together)."""
    _, counts = _grouped_counts(records)
    rng = np.random.default_rng(seed)
    point = fn(_as_counts(counts.sum(0)))
    stats = [
        fn(_as_counts(counts[rng.integers(0, len(counts), len(counts))].sum(0)))
        for _ in range(n_boot)
    ]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "low": float(lo), "high": float(hi)}


def _paired_group_counts(a: Sequence[dict], b: Sequence[dict]):
    keys_a, counts_a = _grouped_counts(a)
    keys_b, counts_b = _grouped_counts(b)
    if keys_a != keys_b:
        raise ValueError("Systems must be scored on the same examples in the same order.")
    return counts_a, counts_b


def bootstrap_diff(
    a: Sequence[dict],
    b: Sequence[dict],
    fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """CI of fn(a) - fn(b) with the same resampled groups for both systems."""
    ca, cb = _paired_group_counts(a, b)
    rng = np.random.default_rng(seed)
    point = fn(_as_counts(ca.sum(0))) - fn(_as_counts(cb.sum(0)))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ca), len(ca))
        diffs.append(fn(_as_counts(ca[idx].sum(0))) - fn(_as_counts(cb[idx].sum(0))))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": point, "low": float(lo), "high": float(hi)}


def permutation_test(
    a: Sequence[dict],
    b: Sequence[dict],
    fn: MetricFn,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired two-sided permutation test: swap the systems' outputs per group at random."""
    ca, cb = _paired_group_counts(a, b)
    rng = np.random.default_rng(seed)
    observed = fn(_as_counts(ca.sum(0))) - fn(_as_counts(cb.sum(0)))
    hits = 0
    for _ in range(n_perm):
        swap = (rng.random(len(ca)) < 0.5)[:, None]
        xa = np.where(swap, cb, ca).sum(0)
        xb = np.where(swap, ca, cb).sum(0)
        hits += abs(fn(_as_counts(xa)) - fn(_as_counts(xb))) >= abs(observed) - 1e-12
    return {"diff": observed, "p_value": (hits + 1) / (n_perm + 1)}
