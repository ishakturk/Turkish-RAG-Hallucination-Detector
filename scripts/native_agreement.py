"""Human validation of the native-set LLM labels (analysis plan §10, amendment 1).

Reads the three label sets in data/, writes results/native_agreement.json and the adjudication
sheet data/native_adjudication_blank.csv (rows where two label sets disagree).

    python scripts/native_agreement.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hdtr import native as N  # noqa: E402

DATA = ROOT / "data"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    sample = json.loads((DATA / "native_human_sample.json").read_text(encoding="utf-8"))
    fp = json.loads((DATA / "native_claude_labels.sha256.json").read_text(encoding="utf-8"))
    claude_bytes = (DATA / "native_labels_claude.csv").read_bytes()
    assert hashlib.sha256(claude_bytes).hexdigest() == fp["labels_sha256"], "Claude labels changed"
    assert hashlib.sha256((DATA / "native_human_sample_blank.csv").read_bytes()).hexdigest() == sample["sheet_sha256"]
    assert hashlib.sha256((DATA / "native_labeling_blank.csv").read_bytes()).hexdigest() == fp["blank_sheet_sha256"]

    claude_df, sheets_df, human_df = (read("native_labels_claude.csv"), read("native_labels_claude_sheets.csv"),
                                      read("native_human_sample_labels.csv"))
    blank = read("native_human_sample_blank.csv").set_index("id")
    all_ids = claude_df["id"].tolist()
    claude, p1 = N.read_label_sheet(claude_df, all_ids)
    sheets, p2 = N.read_label_sheet(sheets_df, all_ids)
    human, p3 = N.read_label_sheet(human_df, sample["ids"])
    assert not p1 and not p2, (p1, p2)
    reference = {i: claude[i] for i in all_ids if claude[i] == sheets[i]}
    llm_disagree = sorted(i for i in all_ids if claude[i] != sheets[i])

    groups = {i: blank.loc[i, "context"] for i in sample["ids"]}
    ids = sample["ids"]
    domains = {"all": ids, "wikipedia": [i for i in ids if i.startswith("wiki")],
               "legal": [i for i in ids if i.startswith("legal")]}
    out = {
        "note": "Human (user, alone, no AI) vs reference LLM labels on the blind 50-row sample, before any "
                "adjudication. Reference = label where both Claude label sets agree. Bootstrap resamples "
                "contexts (2,000 replicates, seed 482). PABAK is supplementary (κ is low when positives are rare).",
        "human_problems": p3,
        "human_vs_reference": {d: N.agreement(human, reference, s, groups) for d, s in domains.items()},
        "human_vs_claude": N.agreement(human, claude, ids, groups),
        "human_vs_claude_sheets": N.agreement(human, sheets, ids, groups),
        "claude_vs_claude_sheets_all_184": N.agreement(claude, sheets, all_ids, {i: i for i in all_ids}),
        "llm_disagreements": llm_disagree,
        "llm_disagreements_in_sample": [i for i in llm_disagree if i in ids],
        "reference_positive_rate_184": sum(reference.values()) / len(reference),
    }
    (ROOT / "results" / "native_agreement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                                           encoding="utf-8", newline="\n")

    # Adjudication sheet: LLM-vs-LLM disagreements, then human-vs-reference disagreements.
    notes = {name: df.set_index("id")["note"] for name, df in
             (("claude", claude_df), ("sheets", sheets_df), ("human", human_df))}
    full = read("native_labeling_blank.csv").set_index("id")
    rows = []
    for i in llm_disagree:
        rows.append({"id": i, "kind": "claude_vs_claude", "label_a": claude[i], "note_a": notes["claude"][i],
                     "label_b": sheets[i], "note_b": notes["sheets"][i]})
    for i in out["human_vs_reference"]["all"]["disagreements"]:
        rows.append({"id": i, "kind": "human_vs_reference", "label_a": human[i], "note_a": notes["human"][i],
                     "label_b": reference[i], "note_b": notes["claude"][i]})
    sheet = pd.DataFrame(rows)
    sheet.insert(0, "no", range(1, len(sheet) + 1))
    for col in ("question", "context", "answer"):
        sheet[col] = [full.loc[i, col] for i in sheet["id"]]
    sheet["final"], sheet["final_note"] = "", ""
    sheet = sheet[["no", "id", "kind", "question", "context", "answer", "label_a", "note_a", "label_b", "note_b",
                   "final", "final_note"]]
    sheet.to_csv(DATA / "native_adjudication_blank.csv", index=False, lineterminator="\n", encoding="utf-8")

    for d, r in out["human_vs_reference"].items():
        print(f"{d:10s} n={r['n']:2d} agree={r['observed_agreement']:.3f} kappa={r['kappa']:.3f} "
              f"[{r['kappa_low']:.3f}, {r['kappa_high']:.3f}] pabak={r['pabak']:.3f} "
              f"human+={r['rater_positive']} ref+={r['other_positive']}")
    print("adjudication rows:", len(sheet))


if __name__ == "__main__":
    main()
