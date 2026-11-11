import numpy as np
import pandas as pd
import pytest

from hdtr import native_eval as NE
from hdtr import reproduce as R
from hdtr.final import mean_sd, records_at, run_scores


def _native():
    test = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "domain": ["wikipedia", "wikipedia", "legal", "legal"],
        "generator": ["qwen", "deepseek", "qwen", "deepseek"],
        "question_type": ["partial", "original", "partial", "original"],
        "question": ["Q1", "Q2", "Q3", "Q4"],
        "passages": [["P1"], ["P1"], ["P2"], ["P3"]], "prompt": ["p1", "p2", "p3", "p4"],
        "answer": ["Lig 1929'da kuruldu.", "Lig eskidir.", "Madde 3 ve 5 uygulanır.", "Ceza verilir."]})
    ref = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": [1, 0, 1, 0]})
    spans = {"a": [{"start": 4, "end": 11, "text": "1929'da"}], "c": [{"start": 8, "end": 13, "text": "ve 5 "}]}
    return NE.native_frame(test, ref, spans, dict(zip(test["id"], test["answer"])))


def test_native_jsonl_roundtrip_matches_notebook_frame(tmp_path):
    frame = _native()
    path = tmp_path / "native.jsonl"
    R.write_jsonl(R.native_records(frame), path)
    back = R.load_native_test(path)
    assert list(back.index) == list(frame.index)
    for col in ("answer", "labels", "reference_label", "task_type", "source_id"):
        assert list(back[col]) == list(frame[col]), col
    assert back.loc["a", "spans"][0]["text"] == "1929'da"


def test_load_native_test_rejects_bad_offsets(tmp_path):
    rows = R.native_records(_native())
    rows[0]["spans"][0]["start"] += 1
    path = tmp_path / "bad.jsonl"
    R.write_jsonl(rows, path)
    with pytest.raises(ValueError):
        R.load_native_test(path)


def _probs(frame, seed):
    rng = np.random.default_rng(seed)
    return [rng.random(len(a)).astype(np.float16).astype(np.float32) for a in frame["answer"]]


def test_prediction_pack_roundtrip(tmp_path):
    frame = _native()
    probs = {"r2_s1": _probs(frame, 1), "sys_x": _probs(frame, 2)}
    np.savez_compressed(tmp_path / "p.npz", **R.pack_predictions(list(frame.index), probs))
    back = R.load_native_predictions(tmp_path / "p.npz", frame)
    assert set(back) == {"r2_s1", "sys_x"}
    for name in probs:
        assert all(np.array_equal(a, b) for a, b in zip(probs[name], back[name]))
    with pytest.raises(ValueError):
        R.load_native_predictions(tmp_path / "p.npz", frame.iloc[::-1])


def test_rescore_averages_seeds_like_notebook_10():
    frame = _native()
    systems = {"r2_s1": {"kind": "ours", "group": "r2", "tau_example": 0.5, "tau_span": 0.6},
               "r2_s2": {"kind": "ours", "group": "r2", "tau_example": 0.4, "tau_span": 0.5},
               "base": {"kind": "existing", "tau_example": 0.3, "tau_span": 0.3}}
    probs = {"r2_s1": _probs(frame, 1), "r2_s2": _probs(frame, 2), "base": _probs(frame, 3)}
    out = R.rescore(frame, probs, systems)
    assert set(out["units"]) == {"r2", "base"}
    direct = mean_sd([run_scores(records_at(frame, probs[n], systems[n])) for n in ("r2_s1", "r2_s2")])
    assert out["units"]["r2"]["tuned"]["overall"]["example_f1"] == direct["tuned"]["overall"]["example_f1"]
    saved = {"systems": {u: {"scores": s} for u, s in out["units"].items()}}
    table = R.compare_with_saved(out, saved)
    assert table["abs_diff"].max() == 0


def test_highlight_marks_predicted_and_gold():
    out = R.highlight_html("a <b> c", predicted=[{"start": 2, "end": 5}], gold=[{"start": 4, "end": 7}])
    assert "&lt;b" in out and "background" in out and "underline" in out
    assert R.highlight_html("plain") == "plain"
