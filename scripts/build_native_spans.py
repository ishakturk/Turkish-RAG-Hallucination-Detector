"""Turn the proposed span texts into character offsets: data/native_spans.json.

Every label-1 row must have at least one span, and no label-0 row may have one.

    python scripts/build_native_spans.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hdtr import native as N  # noqa: E402

DATA = ROOT / "data"


def main() -> None:
    proposed = json.loads((ROOT / "scripts" / "native_spans_proposed.json").read_text(encoding="utf-8"))["spans"]
    ref = pd.read_csv(DATA / "native_reference_labels.csv", dtype={"label": int})
    answers = pd.read_csv(DATA / "native_labeling_blank.csv", dtype=str, keep_default_na=False,
                          encoding="utf-8-sig").set_index("id")["answer"]
    positive = set(ref.loc[ref["label"] == 1, "id"])
    assert set(proposed) == positive, (sorted(positive - set(proposed)), sorted(set(proposed) - positive))
    spans = {i: N.locate_spans(answers[i], proposed[i]) for i in ref["id"] if i in positive}
    out = {"note": "Gold hallucinated spans (character offsets into the answer) for the native test set; "
                   "Claude-proposed, not edited row by row (analysis plan §10, amendment 1). Rows absent here have no span.",
           "spans": spans}
    (DATA / "native_spans.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8", newline="\n")
    chars = sum(s["end"] - s["start"] for v in spans.values() for s in v)
    print(f"{len(spans)} rows, {sum(map(len, spans.values()))} spans, {chars} characters")


if __name__ == "__main__":
    main()
