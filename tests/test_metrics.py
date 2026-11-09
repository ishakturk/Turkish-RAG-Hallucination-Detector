import numpy as np
import pytest

from hdtr.metrics import bootstrap_ci, bootstrap_diff, evaluate, metric, permutation_test

ANSWER = "Son başvuru 25 Ocak'a kadar."
GOLD = [{"start": 12, "end": 21}]


def _rec(pred, gold=GOLD, task="QA", group=0, probs=None):
    return {"answer": ANSWER, "gold_spans": gold, "pred_spans": pred,
            "task_type": task, "group": group, "char_probs": probs}


def test_perfect_prediction():
    m = evaluate([_rec(GOLD), _rec([], gold=[], group=1)])["overall"]
    assert m["example"]["f1"] == 1.0
    assert m["word"]["f1"] == 1.0
    assert m["span"]["f1"] == 1.0
    assert m["example"]["confusion"] == [[1, 0], [0, 1]]


def test_partial_span_overlap():
    pred = [{"start": 15, "end": 21}]  # "Ocak'a" only
    m = evaluate([_rec(pred)])["overall"]
    assert m["span"]["precision"] == 1.0
    assert m["span"]["recall"] == pytest.approx(6 / 9)
    assert m["word"]["recall"] == 0.5  # 1 of 2 gold words
    assert m["example"]["f1"] == 1.0


def test_false_positive_on_clean_answer():
    m = evaluate([_rec([{"start": 0, "end": 3}], gold=[])])["overall"]
    assert m["example"]["precision"] == 0.0
    assert m["example"]["confusion"] == [[0, 1], [0, 0]]


def test_per_task_breakdown_and_auroc():
    hi = np.full(len(ANSWER), 0.9, dtype=np.float32)
    lo = np.full(len(ANSWER), 0.1, dtype=np.float32)
    recs = [_rec(GOLD, task="QA", probs=hi), _rec([], gold=[], task="Summary", group=1, probs=lo)]
    result = evaluate(recs)
    assert set(result) == {"overall", "QA", "Summary"}
    assert result["overall"]["example_auroc"] == 1.0


def _system(correct_share, n=60, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        gold = GOLD if i % 2 == 0 else []
        pred = gold if rng.random() < correct_share else (GOLD if not gold else [])
        recs.append(_rec(pred, gold=gold, group=i // 3))
    return recs


def test_bootstrap_ci_contains_point_and_is_reproducible():
    recs = _system(0.8)
    a = bootstrap_ci(recs, metric("example"), n_boot=300, seed=1)
    b = bootstrap_ci(recs, metric("example"), n_boot=300, seed=1)
    assert a == b
    assert a["low"] <= a["point"] <= a["high"]


def test_paired_tests_detect_clear_difference():
    good, bad = _system(0.95, seed=1), _system(0.4, seed=2)
    diff = bootstrap_diff(good, bad, metric("example"), n_boot=300)
    assert diff["low"] > 0
    assert permutation_test(good, bad, metric("example"), n_perm=500)["p_value"] < 0.05


def test_paired_tests_require_same_groups():
    with pytest.raises(ValueError):
        bootstrap_diff(_system(0.8, n=6), _system(0.8, n=9), metric("example"))


def test_bootstrap_diff_seeds_matches_single_run_and_averages():
    from hdtr.metrics import bootstrap_diff, bootstrap_diff_seeds, metric

    def rec(group, gold, pred):
        a = "Lig 1927 yılında kuruldu."
        return {"answer": a, "group": group, "gold_spans": [{"start": 4, "end": 8}] if gold else [],
                "pred_spans": [{"start": 4, "end": 8}] if pred else []}

    good = [rec(g, g % 2 == 0, g % 2 == 0) for g in range(20)]
    bad = [rec(g, g % 2 == 0, True) for g in range(20)]
    fn = metric("example", "f1")
    single = bootstrap_diff(good, bad, fn, n_boot=200)
    multi = bootstrap_diff_seeds([good], [bad], fn, n_boot=200)
    assert abs(single["diff"] - multi["diff"]) < 1e-12 and abs(single["low"] - multi["low"]) < 1e-12
    two = bootstrap_diff_seeds([good, bad], [bad], fn, n_boot=200)
    assert abs(two["a_mean"] - (1.0 + fn_val(bad, fn)) / 2) < 1e-9


def fn_val(records, fn):
    from hdtr.metrics import record_counts, sum_counts
    return fn(sum_counts(record_counts(r) for r in records))
