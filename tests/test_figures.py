import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from hdtr import figures as F  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_build_all_writes_every_figure_and_table(tmp_path):
    written = F.build_all(_load("06_train.json"), _load("07_thresholds.json"), _load("08_final_test.json"), tmp_path)
    assert {k for k in written if k.startswith("fig")} == {
        "fig1_main_results", "fig2_q1_q5_forest", "fig3_span_f1_per_task", "fig4_threshold_curves",
        "fig5_training_curves"}
    assert all(Path(p).stat().st_size > 5000 for k, paths in written.items() if k.startswith("fig") for p in paths)
    assert all(Path(p).stat().st_size > 200 for k, paths in written.items() if k.startswith("table") for p in paths)


def test_tables_follow_the_08_numbers():
    final = _load("08_final_test.json")
    main = F.main_table(final)
    assert len(main) == len(final["systems"]["full"]) and main.loc[0, "System"] == F.NAMES[final["decisions"]["final_model"]]
    q = F.comparisons_table(final)
    q1 = q[(q.Q == "Q1") & (q.Metric == "Span F1")].iloc[0]
    assert q1.Role == "primary" and q1["CI low"] > 0
    assert set(q.Q) == {"Q1", "Q2", "Q3", "Q4", "Q5"}


def test_native_figures_and_turkish_labels(tmp_path):
    args = (_load("06_train.json"), _load("07_thresholds.json"), _load("08_final_test.json"))
    native = _load("10_native_test.json")
    written = F.build_all(*args, tmp_path / "tr", native=native, lang="tr")
    assert {"fig6_translated_vs_native", "fig7_native_rates", "table4_native"} <= set(written)
    assert F.name("r2") == F.NAMES["r2"]                                  # language reset afterwards
    table = F.native_table(native)
    assert len(table) == len(native["systems"]) and table.loc[0, "System"] == F.NAMES["r4"]
