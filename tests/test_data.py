import pandas as pd

from hdtr.data import align_by_order, audit_row, drop_decision, make_splits


def test_audit_clean_row_has_no_drop():
    flags = audit_row("Başvurular 25 Ocak'a kadar sürer.", [{"start": 11, "end": 20}],
                      "Applications last until 25 January.", [{"start": 24, "end": 34}])
    assert not drop_decision(flags)


def test_audit_flags_truncation_and_span_problems():
    flags = audit_row("Başvurular 25 Ocak", [{"start": 30, "end": 40}],
                      "Applications last until 25 January.", [])
    assert flags["truncated"] and flags["span_out_of_bounds"] and flags["span_count_mismatch"]
    assert drop_decision(flags)


def test_audit_detects_english_residue_and_repetition():
    eng = audit_row("The applications are open for all the students in the city.", [])
    assert eng["english_residue"] and not drop_decision(eng)  # soft flag alone keeps the row
    rep = audit_row("bir iki üç dört " * 7, [])
    assert rep["repetition"] and drop_decision(rep)
    # A name repeated just as often in the English original is not degenerate.
    name = "Su Casa Fresh Mexican sunar. " * 6
    assert not audit_row(name, [], "Su Casa Fresh Mexican offers. " * 6, [])["repetition"]


def test_number_added_by_translation_is_flagged():
    en = "Preheat the oven to 350 degrees Fahrenheit."
    tr = "Fırını 175 dereceye (350 Fahrenheit) ısıtın."
    flags = audit_row(tr, [], en, [], prompt="Bake at 350 F.")
    assert flags["number_added"] and not drop_decision(flags)
    same = audit_row("Fırını 350 Fahrenheit dereceye ısıtın.", [], en, [])
    assert not same["number_added"]


def test_align_by_order_within_groups():
    tr = pd.DataFrame({"row_id": [0, 1], "split": ["train", "test"], "task_type": ["QA", "QA"],
                       "answer": ["Cevap 2024 yılında.", "Yanıt 15."],
                       "labels": [[], [{"start": 0, "end": 5}]]})
    en = pd.DataFrame({"response_id": ["a", "b"], "source_id": ["s1", "s2"], "split": ["train", "test"],
                       "task_type": ["QA", "QA"], "answer": ["Answer in 2024.", "Reply 15."],
                       "labels": [[], [{"start": 0, "end": 5}]]})
    signals = align_by_order(tr, en, by=("split", "task_type"))
    assert signals["pass"].all()
    assert set(signals["source_id"]) == {"s1", "s2"}


def test_make_splits_keeps_sources_together():
    rows = []
    for split, n_src in (("train", 40), ("test", 40)):
        for task in ("QA", "Summary"):
            for s in range(n_src):
                for k in range(6):
                    rows.append({"split": split, "task_type": task, "source_id": f"{split}-{task}-{s}"})
    df = pd.DataFrame(rows)
    df["row_id"] = range(len(df))
    splits = make_splits(df, dev_sources_per_task=10, val_frac=0.1)
    assert len(splits["dev"]) == 2 * 10 * 6
    assert sum(len(v) for v in splits.values()) == len(df)
    src = df.set_index("row_id")["source_id"]
    seen = {}
    for name, ids in splits.items():
        for sid in src.loc[ids].unique():
            assert seen.setdefault(sid, name) == name
