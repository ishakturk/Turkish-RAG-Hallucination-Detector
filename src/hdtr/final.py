"""Threshold freezing (notebook 07) and the one-time final-test scoring (notebook 08).

Both notebooks follow docs/analysis-plan.md §4–5:
- thresholds are tuned on dev for every system and frozen before the final test is touched;
- on the final test, example metrics use τ_example and word/span metrics use τ_span;
- uncertainty is a source-grouped bootstrap of the seed-averaged metric (2,000 replicates).
"""

from __future__ import annotations

import hashlib
import pickle
import statistics
from typing import Sequence

import numpy as np

from .evaluation import compact, make_records, tune_thresholds
from .metrics import bootstrap_ci_seeds, bootstrap_diff_seeds, evaluate, metric, permutation_test

# Existing detectors compete for "best baseline"; classical baselines are reported but are not candidates.
EXISTING = ("newmind_modernbert_tr", "newmind_turkembed4sts", "newmind_eurobert_tr", "newmind_bge_m3",
            "newmind_ettin_150m", "newmind_ettin_32m", "krlabs_v2_mmbert_zeroshot")
CLASSICAL = ("classical_lexical_overlap", "classical_word_logreg")
OUR_GROUPS = ("r0", "r1", "r2", "r3", "r4")
PARAMS_M = {"r1": 149, "r2": 307, "r3": 307}
TIE_MARGIN = 0.01

# (id, label, a, b): the reported effect is a − b. "final", "best_baseline" and "r4_base" are resolved
# from the frozen decisions in 07_thresholds.json.
COMPARISONS = (
    ("Q1", "final model − best baseline", "final", "best_baseline"),
    ("Q2", "R1 (TabiBERT) − R2 (mmBERT-base)", "r1", "r2"),
    ("Q3", "R3 (KR Labs warm start) − R2 (mmBERT-base)", "r3", "r2"),
    ("Q4", "R1 (our recipe) − R0 (Newmind-style recipe)", "r1", "r0"),
    ("Q5", "R4 (with augmentation) − its base", "r4", "r4_base"),
)
# Primary metrics and the frozen threshold each one is computed at.
PRIMARY = {"example_f1": ("tau_example", "example"), "span_f1": ("tau_span", "span")}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_probs(path: str, row_ids: Sequence) -> list[np.ndarray]:
    """Per-character probabilities from a predictions pickle, in the order of ``row_ids``."""
    with open(path, "rb") as f:
        saved = pickle.load(f)
    by_id = dict(zip(saved["row_ids"], saved["probs"]))
    missing = [i for i in row_ids if i not in by_id]
    if missing:
        raise KeyError(f"{path}: {len(missing)} rows have no prediction (first: {missing[0]})")
    return [np.asarray(by_id[i], dtype=np.float32) for i in row_ids]


def _merge_tuned(ev_example: dict, ev_span: dict) -> dict:
    """Example metrics from the τ_example scoring, word and span metrics from the τ_span scoring.
    AUROC does not depend on the threshold."""
    out = {}
    for key in ev_span:
        ex, sp = compact(ev_example[key]), compact(ev_span[key])
        out[key] = {**sp, **{k: v for k, v in ex.items() if k.startswith("example_")}}
    return out


def tune_system(dev_frame, probs) -> dict:
    """Dev thresholds and dev scores for one system (one run)."""
    tuned = tune_thresholds(dev_frame, probs)
    recs = records_at(dev_frame, probs, tuned)
    ev = {k: evaluate(r) for k, r in recs.items()}
    dev = {"tau_0.5": {k: compact(m) for k, m in ev["tau_0.5"].items()},
           "tuned": _merge_tuned(ev["tau_example"], ev["tau_span"])}
    return {"tau_example": tuned["tau_example"], "tau_span": tuned["tau_span"], "curve": tuned["curve"],
            "selection_score": selection_score(dev), "dev": dev}


def selection_score(dev: dict) -> float:
    """(example F1 at τ_example + span F1 at τ_span) / 2, overall (analysis plan §4.3)."""
    o = dev["tuned"]["overall"]
    return (o["example_f1"] + o["span_f1"]) / 2


def apply_rules(group_scores: dict, baseline_scores: dict) -> dict:
    """R4 base, final model and best baseline from dev selection scores (analysis plan §4.3–4.5).

    ``group_scores`` maps an our-run group to the list of its seeds' selection scores;
    ``baseline_scores`` maps an existing detector to its selection score."""
    means = {g: float(np.mean(v)) for g, v in group_scores.items() if v}
    ranked = sorted((g for g in ("r1", "r2", "r3") if g in means), key=means.get, reverse=True)
    r4_base = ranked[0] if ranked else None
    for g in ranked[1:]:
        if means[r4_base] - means[g] < TIE_MARGIN and PARAMS_M[g] < PARAMS_M[r4_base]:
            r4_base = g
    final = None
    if r4_base:
        final = "r4" if "r4" in means and means["r4"] >= means[r4_base] else r4_base
    candidates = {k: v for k, v in baseline_scores.items() if k in EXISTING}
    best_baseline = max(candidates, key=candidates.get) if candidates else None
    return {"group_means": means, "r4_base": r4_base, "final_model": final,
            "best_baseline": best_baseline, "baseline_scores": candidates}


def records_at(frame, probs, thresholds: dict) -> dict:
    """Metric records at τ = 0.5, τ_example and τ_span (shared when the thresholds coincide)."""
    cache: dict = {}

    def at(th):
        if th not in cache:
            cache[th] = make_records(frame, probs, th)
        return cache[th]

    return {"tau_0.5": at(0.5), "tau_example": at(thresholds["tau_example"]),
            "tau_span": at(thresholds["tau_span"])}


def subset(records_by_tau: dict, keep: Sequence[bool]) -> dict:
    return {k: [r for r, m in zip(recs, keep) if m] for k, recs in records_by_tau.items()}


def run_scores(records_by_tau: dict) -> dict:
    """Scores of one run: at τ = 0.5 and at the frozen thresholds, overall and per task."""
    ev = {k: evaluate(r) for k, r in records_by_tau.items()}
    return {"tau_0.5": {k: compact(m) for k, m in ev["tau_0.5"].items()},
            "tuned": _merge_tuned(ev["tau_example"], ev["tau_span"])}


def mean_sd(run_score_list: Sequence[dict]) -> dict:
    """Mean and sd over runs (seeds) for every number in ``run_scores`` output."""
    out: dict = {}
    for tau, by_key in run_score_list[0].items():
        out[tau] = {}
        for key, metrics in by_key.items():
            out[tau][key] = {}
            for name in metrics:
                vals = [r[tau][key][name] for r in run_score_list if r[tau].get(key, {}).get(name) is not None]
                if not vals:
                    continue
                out[tau][key][name] = {"mean": float(np.mean(vals)),
                                       "sd": float(statistics.stdev(vals)) if len(vals) > 1 else None,
                                       "runs": [float(v) for v in vals]}
    return out


def system_cis(runs: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of each primary metric, seed-averaged, resampling sources."""
    return {name: bootstrap_ci_seeds([r[tau] for r in runs], metric(level), n_boot=n_boot, seed=seed)
            for name, (tau, level) in PRIMARY.items()}


def compare(a_runs: Sequence[dict], b_runs: Sequence[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Seed-averaged difference a − b for each primary metric with its 95% bootstrap CI."""
    return {name: bootstrap_diff_seeds([r[tau] for r in a_runs], [r[tau] for r in b_runs], metric(level),
                                       n_boot=n_boot, seed=seed)
            for name, (tau, level) in PRIMARY.items()}


def permutation_per_seed(a_runs: Sequence[dict], b_run: dict, n_perm: int = 10000, seed: int = 0) -> dict:
    """Paired permutation test of each seed of ``a`` against the single run ``b`` (analysis plan §5)."""
    return {name: [permutation_test(r[tau], b_run[tau], metric(level), n_perm=n_perm, seed=seed)
                   for r in a_runs]
            for name, (tau, level) in PRIMARY.items()}


def q1_verdict(diff: dict, perms: dict) -> dict:
    """Per primary metric: 'better' if the CI excludes 0 on the positive side; robust if every seed has p < 0.05."""
    return {name: {"better": diff[name]["low"] > 0,
                   "robust_all_seeds_p<0.05": all(p["p_value"] < 0.05 for p in perms[name]),
                   "diff": diff[name]["diff"], "ci": [diff[name]["low"], diff[name]["high"]]}
            for name in PRIMARY}
