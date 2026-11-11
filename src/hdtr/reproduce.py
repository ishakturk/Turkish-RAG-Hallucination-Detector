"""Helpers for the deliverable notebook (main.ipynb) and the release notebook (11).

- The native test set travels as ``data/native_test.jsonl`` (one answer per line, with its label and
  gold spans) and every system's native predictions as ``data/native_predictions.npz``, so the native
  results can be recomputed without a GPU or Google Drive.
- ``detect`` runs a detector on one (question, passages, answer) triple for the live demo.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .augment import qa_prompt
from .final import mean_sd, records_at, run_scores
from .native_eval import context_key
from .postprocess import char_probs_to_spans

NATIVE_FILE = "data/native_test.jsonl"
NATIVE_PREDICTIONS = "data/native_predictions.npz"
# The released checkpoint: the R2 seed with the highest dev selection score in results/07_thresholds.json
# (chosen on dev only, before looking at any test number of the individual seeds).
RELEASE_RUN = "r2_mmbert_s42"
NATIVE_COLUMNS = ("id", "domain", "generator", "question_type", "question", "passages", "prompt", "answer",
                  "label", "spans")


# --------------------------------------------------------------------------------------------
# Native test set: export (notebook 11) and load (main.ipynb)
# --------------------------------------------------------------------------------------------

def native_records(frame: pd.DataFrame) -> list[dict]:
    """JSON-ready rows of a frame built by ``native_eval.native_frame``."""
    rows = []
    for i, r in frame.iterrows():
        spans = [{"start": int(s["start"]), "end": int(s["end"]), "text": r["answer"][s["start"]:s["end"]]}
                 for s in r["labels"]]
        rows.append({"id": str(i), "domain": r["domain"], "generator": r["generator"],
                     "question_type": r["question_type"], "question": r["question"],
                     "passages": [str(p) for p in r["passages"]], "prompt": r["prompt"], "answer": r["answer"],
                     "label": int(r["reference_label"]), "spans": spans})
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_native_test(path: str | Path) -> pd.DataFrame:
    """The native set in the shape notebook 10 scored (labels, reference_label, task_type, source_id)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    frame = pd.DataFrame(rows).set_index("id", drop=False)
    for i, r in frame.iterrows():
        if bool(r["spans"]) != bool(r["label"]):
            raise ValueError(f"{i}: label {r['label']} but {len(r['spans'])} spans")
        for s in r["spans"]:
            if r["answer"][s["start"]:s["end"]] != s["text"]:
                raise ValueError(f"{i}: span offsets do not match the answer text")
    frame["labels"] = [[{"start": s["start"], "end": s["end"], "label": "native"} for s in spans]
                       for spans in frame["spans"]]
    frame["reference_label"] = frame["label"].astype(int)
    frame["task_type"] = frame["domain"]
    frame["source_id"] = [context_key(p) for p in frame["passages"]]
    return frame


# --------------------------------------------------------------------------------------------
# Native predictions: one compressed file for all systems
# --------------------------------------------------------------------------------------------

def pack_predictions(ids: Sequence[str], probs_by_system: dict[str, Sequence[np.ndarray]]) -> dict[str, np.ndarray]:
    """Arrays for ``np.savez_compressed``: ids, per-answer lengths, and each system's character
    probabilities concatenated in ``ids`` order (float16, as notebook 10 saved them)."""
    lengths = None
    out: dict[str, np.ndarray] = {"ids": np.array(list(ids))}
    for name, probs in probs_by_system.items():
        lens = np.array([len(p) for p in probs], dtype=np.int64)
        if lengths is None:
            lengths = lens
        elif not np.array_equal(lengths, lens):
            raise ValueError(f"{name}: answer lengths differ from the other systems")
        out[f"p::{name}"] = np.concatenate([np.asarray(p, dtype=np.float16) for p in probs])
    out["lengths"] = lengths if lengths is not None else np.zeros(0, dtype=np.int64)
    return out


def unpack_predictions(arrays) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    ids = [str(i) for i in arrays["ids"]]
    cuts = np.cumsum(arrays["lengths"])[:-1]
    systems = {k[3:]: [p.astype(np.float32) for p in np.split(arrays[k], cuts)]
               for k in arrays.keys() if k.startswith("p::")}
    return ids, systems


def load_native_predictions(path: str | Path, frame: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    with np.load(path) as arrays:
        ids, systems = unpack_predictions(arrays)
    if ids != [str(i) for i in frame.index]:
        raise ValueError("prediction ids are not in the order of the native test file")
    for name, probs in systems.items():
        bad = [i for i, p, a in zip(ids, probs, frame["answer"]) if len(p) != len(a)]
        if bad:
            raise ValueError(f"{name}: prediction length differs from the answer for {bad[:3]}")
    return systems


# --------------------------------------------------------------------------------------------
# Rescoring with the frozen thresholds (same functions as notebook 10)
# --------------------------------------------------------------------------------------------

def unit_of(name: str, system: dict) -> str:
    return system["group"] if system["kind"] == "ours" else name


def rescore(frame: pd.DataFrame, probs_by_system: dict[str, list[np.ndarray]], systems: dict) -> dict:
    """Seed-averaged scores per unit (R0–R4 groups and each existing/classical system)."""
    per_run = {name: run_scores(records_at(frame, probs, systems[name])) for name, probs in probs_by_system.items()}
    units: dict[str, list[str]] = {}
    for name in per_run:
        units.setdefault(unit_of(name, systems[name]), []).append(name)
    return {"per_run": per_run, "units": {u: mean_sd([per_run[n] for n in runs]) for u, runs in units.items()}}


def compare_with_saved(rescored: dict, saved: dict,
                       metrics: Sequence[str] = ("example_f1", "span_f1", "example_auroc")) -> pd.DataFrame:
    """Recomputed vs saved (results/10_native_test.json) seed-mean scores at the frozen thresholds."""
    rows = []
    for unit, scores in rescored["units"].items():
        mine = scores["tuned"]["overall"]
        theirs = saved["systems"][unit]["scores"]["tuned"]["overall"]
        for m in metrics:
            rows.append({"system": unit, "metric": m, "recomputed": mine[m]["mean"], "saved": theirs[m]["mean"],
                         "abs_diff": abs(mine[m]["mean"] - theirs[m]["mean"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Live demo
# --------------------------------------------------------------------------------------------

def detect(model, tokenizer, question: str, passages: Sequence[str], answer: str, threshold: float,
           max_length: int = 4096, variant: int = 0, prompt: str | None = None) -> dict:
    """Spans of ``answer`` the detector marks as unsupported, with the answer's maximum probability."""
    from .infer import predict_char_probs

    prompt = prompt if prompt is not None else qa_prompt(question, list(passages), variant=variant)
    probs, _ = predict_char_probs(model, tokenizer, prompt, answer, max_length=max_length)
    return {"spans": char_probs_to_spans(answer, probs, threshold), "max_prob": float(probs.max()) if len(probs) else 0.0,
            "probs": probs}


def highlight_html(answer: str, predicted: Sequence[dict] = (), gold: Sequence[dict] = ()) -> str:
    """Answer text with predicted spans in red and gold spans underlined (for IPython.display.HTML)."""
    marks = np.zeros((len(answer), 2), dtype=bool)
    for s in predicted:
        marks[s["start"]:s["end"], 0] = True
    for s in gold:
        marks[s["start"]:s["end"], 1] = True
    out, k = [], 0
    while k < len(answer):
        j = k
        while j < len(answer) and (marks[j] == marks[k]).all():
            j += 1
        piece = html.escape(answer[k:j])
        pred, is_gold = marks[k]
        style = []
        if pred:
            style.append("background:#f8c9b8;color:#7a1f00")
        if is_gold:
            style.append("text-decoration:underline;text-decoration-thickness:2px")
        out.append(f'<span style="{";".join(style)}">{piece}</span>' if style else piece)
        k = j
    return "".join(out)
