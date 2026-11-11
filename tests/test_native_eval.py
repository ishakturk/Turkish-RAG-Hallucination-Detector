import numpy as np
import pandas as pd
import pytest

from hdtr import native_eval as NE
from hdtr.evaluation import make_records


def _test_frame():
    return pd.DataFrame({
        "id": ["a", "b", "c"], "domain": ["wikipedia", "wikipedia", "legal"],
        "passages": [["P1"], ["P1"], ["P2"]],
        "answer": ["Lig 1929'da kuruldu.", "Lig eskidir.", "Madde 3 uygulanır."]})


def test_native_frame_attaches_spans_and_checks_consistency():
    test = _test_frame()
    ref = pd.DataFrame({"id": ["a", "b", "c"], "label": [1, 0, 0]})
    spans = {"a": [{"start": 4, "end": 11, "text": "1929'da"}]}
    answers = dict(zip(test["id"], test["answer"]))
    f = NE.native_frame(test, ref, spans, answers)
    assert f.loc["a", "labels"] == [{"start": 4, "end": 11, "label": "native"}] and f.loc["b", "labels"] == []
    assert f.loc["a", "source_id"] == f.loc["b", "source_id"] != f.loc["c", "source_id"]
    assert list(f["task_type"]) == ["wikipedia", "wikipedia", "legal"]
    with pytest.raises(ValueError):                     # label 1 without a span
        NE.native_frame(test, ref.assign(label=[1, 1, 0]), spans, answers)
    with pytest.raises(ValueError):                     # answer changed after labeling
        NE.native_frame(test, ref, spans, {**answers, "c": "başka"})


def test_example_auroc_ci_groups_and_averages_runs():
    test = _test_frame()
    ref = pd.DataFrame({"id": ["a", "b", "c"], "label": [1, 0, 0]})
    f = NE.native_frame(test, ref, {"a": [{"start": 4, "end": 11}]}, dict(zip(test["id"], test["answer"])))
    good = [np.full(len(a), p, dtype=np.float32) for a, p in zip(f["answer"], (0.9, 0.1, 0.2))]
    bad = [np.full(len(a), p, dtype=np.float32) for a, p in zip(f["answer"], (0.1, 0.9, 0.2))]
    runs = [make_records(f, good), make_records(f, bad)]
    out = NE.example_auroc_ci(runs, n_boot=200)
    assert out["point"] == pytest.approx(0.5) and out["n_groups"] == 2    # mean of AUROC 1.0 and 0.0
    assert 0 < out["replicates_used"] <= 200 and out["low"] <= out["high"]
