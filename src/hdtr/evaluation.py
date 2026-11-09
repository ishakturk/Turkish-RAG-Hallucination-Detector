"""Shared scoring helpers for notebooks: records from predictions, dev threshold tuning, and a
compact metrics summary. Every system (baselines and our runs) goes through these, so the
comparison uses one protocol (docs/analysis-plan.md)."""

from __future__ import annotations

import numpy as np

from .metrics import evaluate
from .postprocess import char_probs_to_spans

THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.96, 0.05))


def make_records(frame, probs, threshold: float = 0.5, group_col: str = "source_id") -> list[dict]:
    """One metrics record per row: gold spans, predicted spans at ``threshold``, probabilities."""
    groups = frame[group_col] if group_col in frame else frame.index
    return [{"answer": a, "gold_spans": list(g), "char_probs": p, "task_type": t, "group": s,
             "pred_spans": char_probs_to_spans(a, p, threshold)}
            for a, g, p, t, s in zip(frame["answer"], frame["labels"], probs, frame["task_type"], groups)]


def compact(ev: dict) -> dict:
    """The numbers we report from one ``evaluate`` result (overall or one task)."""
    return {"n": ev.get("n"),
            "example_f1": ev["example"]["f1"], "example_p": ev["example"]["precision"],
            "example_r": ev["example"]["recall"],
            "word_f1": ev["word"]["f1"], "word_macro_f1": ev["word"].get("macro_f1"),
            "span_f1": ev["span"]["f1"], "span_p": ev["span"]["precision"], "span_r": ev["span"]["recall"],
            "example_auroc": ev.get("example_auroc"), "word_auroc": ev.get("word_auroc")}


def score(frame, probs, threshold: float = 0.5, by_task: bool = True) -> dict:
    ev = evaluate(make_records(frame, probs, threshold), by_task=by_task)
    return {key: compact(m) for key, m in ev.items()}


def tune_thresholds(frame, probs, grid=THRESHOLD_GRID) -> dict:
    """τ_example maximises example F1 and τ_span maximises span F1 on this frame (dev only).
    Ties go to the threshold closest to 0.5."""
    curve = {}
    for th in grid:
        m = evaluate(make_records(frame, probs, th), by_task=False)["overall"]
        curve[th] = (m["example"]["f1"], m["span"]["f1"])
    best = lambda k: max(grid, key=lambda th: (round(curve[th][k], 6), -abs(th - 0.5)))
    return {"tau_example": best(0), "tau_span": best(1),
            "curve": {str(th): {"example_f1": e, "span_f1": s} for th, (e, s) in curve.items()}}
