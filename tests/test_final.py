import pickle

import numpy as np
import pandas as pd
import pytest

from hdtr.final import (apply_rules, compare, load_probs, mean_sd, q1_verdict, records_at, run_scores,
                        selection_score, subset, tune_system)
from hdtr.metrics import bootstrap_ci, bootstrap_ci_seeds, metric

ANSWER = "Lig 1927 yılında kuruldu."  # "1927" is characters 4–8


def _frame(n=12):
    gold = [[{"start": 4, "end": 8}] if i % 2 == 0 else [] for i in range(n)]
    return pd.DataFrame({"answer": [ANSWER] * n, "labels": gold, "source_id": [i // 2 for i in range(n)],
                         "task_type": ["QA", "Summary"] * (n // 2)}, index=[100 + i for i in range(n)])


def _probs(frame, hit=0.7, miss=0.0, noise=0.0):
    out = []
    for labels in frame["labels"]:
        p = np.full(len(ANSWER), miss, dtype=np.float32)
        p[4:8] = hit if labels else noise
        out.append(p)
    return out


def test_bootstrap_ci_seeds_equals_single_run_ci():
    frame = _frame()
    recs = records_at(frame, _probs(frame, noise=0.6), {"tau_example": 0.5, "tau_span": 0.5})
    one = bootstrap_ci(recs["tau_0.5"], metric("example"), n_boot=200, seed=3)
    seeds = bootstrap_ci_seeds([recs["tau_0.5"]], metric("example"), n_boot=200, seed=3)
    assert seeds["point"] == pytest.approx(one["point"])
    assert seeds["low"] == pytest.approx(one["low"]) and seeds["high"] == pytest.approx(one["high"])


def test_tuned_scores_take_example_metrics_at_tau_example_and_span_at_tau_span():
    frame = _frame()
    # Gold words get 0.7, clean answers 0.4 on the same word: τ 0.5 separates them, τ 0.3 flags everything.
    probs = _probs(frame, hit=0.7, noise=0.4)
    s = run_scores(records_at(frame, probs, {"tau_example": 0.5, "tau_span": 0.3}))
    assert s["tuned"]["overall"]["example_f1"] == 1.0           # from τ_example = 0.5
    assert s["tuned"]["overall"]["span_f1"] < 1.0               # from τ_span = 0.3 (false spans)
    assert s["tau_0.5"]["overall"]["span_f1"] == 1.0
    assert set(s["tuned"]) == {"overall", "QA", "Summary"}


def test_records_are_shared_when_thresholds_coincide_and_subset_keeps_order():
    frame = _frame()
    recs = records_at(frame, _probs(frame), {"tau_example": 0.5, "tau_span": 0.5})
    assert recs["tau_example"] is recs["tau_0.5"] is recs["tau_span"]
    keep = [i % 3 != 0 for i in range(len(frame))]
    sub = subset(recs, keep)
    assert len(sub["tau_span"]) == sum(keep) and sub["tau_span"][0]["group"] == frame["source_id"].iloc[1]


def test_tune_system_reports_selection_score():
    frame = _frame()
    s = tune_system(frame, _probs(frame, hit=0.3))
    assert s["tau_example"] == 0.3 and s["tau_span"] == 0.3
    assert s["selection_score"] == pytest.approx(1.0) == selection_score(s["dev"])


def test_rules_tie_goes_to_smaller_model_and_r4_needs_at_least_its_base():
    d = apply_rules({"r1": [0.630, 0.630], "r2": [0.638, 0.638], "r3": [0.60], "r4": [0.634]},
                    {"newmind_turkembed4sts": 0.6, "newmind_bge_m3": 0.55, "classical_word_logreg": 0.9})
    assert d["r4_base"] == "r1"                     # 0.008 gap, TabiBERT is smaller
    assert d["final_model"] == "r4"                 # 0.634 ≥ r1's 0.630
    assert d["best_baseline"] == "newmind_turkembed4sts"   # classical baselines are not candidates
    d = apply_rules({"r1": [0.593], "r2": [0.638], "r3": [0.623], "r4": [0.634]}, {})
    assert d["r4_base"] == "r2" and d["final_model"] == "r2" and d["best_baseline"] is None


def test_load_probs_reorders_and_rejects_missing_rows(tmp_path):
    path = tmp_path / "p.pkl"
    with open(path, "wb") as f:
        pickle.dump({"row_ids": [5, 7], "probs": [np.array([0.1], np.float16), np.array([0.9], np.float16)]}, f)
    out = load_probs(str(path), [7, 5])
    assert out[0].dtype == np.float32 and out[0][0] == pytest.approx(0.9, abs=1e-3)
    with pytest.raises(KeyError):
        load_probs(str(path), [5, 6])


def test_compare_and_q1_verdict_on_a_clear_difference():
    frame = _frame(40)
    th = {"tau_example": 0.5, "tau_span": 0.5}
    good = [records_at(frame, _probs(frame), th) for _ in range(2)]
    bad = [records_at(frame, _probs(frame, hit=0.2, noise=0.8), th)]
    diff = compare(good, bad, n_boot=200)
    assert diff["example_f1"]["low"] > 0 and diff["span_f1"]["diff"] == pytest.approx(1.0)
    perms = {k: [{"p_value": 0.001}, {"p_value": 0.2}] for k in diff}
    v = q1_verdict(diff, perms)
    assert v["span_f1"]["better"] and not v["span_f1"]["robust_all_seeds_p<0.05"]


def test_mean_sd_handles_single_run():
    frame = _frame()
    s = run_scores(records_at(frame, _probs(frame), {"tau_example": 0.5, "tau_span": 0.5}))
    one = mean_sd([s])
    assert one["tuned"]["overall"]["span_f1"] == {"mean": 1.0, "sd": None, "runs": [1.0]}
    two = mean_sd([s, s])
    assert two["tuned"]["overall"]["span_f1"]["sd"] == 0.0
