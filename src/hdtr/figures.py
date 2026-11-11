"""Report figures and tables (docs/analysis-plan.md §9) built from the committed result files.

Everything here reads results/06_train.json, results/07_thresholds.json and results/08_final_test.json,
so the figures can be rebuilt anywhere without predictions or a GPU. Figures are sized for an IEEE
two-column paper (3.5 in single column, 7.16 in double column) and saved as PDF (for LaTeX) and PNG.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, text inks, surface.
# Slots 1-3 pass the all-pairs colour-vision check, so kinds use them; up to 5 entities use slots 1-5.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
KIND_COLOR = {"ours": SLOTS[0], "existing": SLOTS[1], "classical": SLOTS[2]}
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

NAMES = {
    "r0": "R0 TabiBERT, Newmind recipe", "r1": "R1 TabiBERT", "r2": "R2 mmBERT-base (final)",
    "r3": "R3 KR Labs warm start", "r4": "R4 mmBERT + augmentation",
    "newmind_modernbert_tr": "ModernBERT-tr", "newmind_turkembed4sts": "TurkEmbed4STS (best baseline)",
    "newmind_eurobert_tr": "EuroBERT-tr", "newmind_bge_m3": "bge-m3", "newmind_ettin_150m": "Ettin-150M",
    "newmind_ettin_32m": "Ettin-32M", "krlabs_v2_mmbert_zeroshot": "KR Labs v2 (zero-shot)",
    "classical_lexical_overlap": "Lexical overlap", "classical_word_logreg": "Word log. regression",
}
METRIC_NAMES = {"example_f1": "Example F1", "span_f1": "Span F1"}
TASKS = ("QA", "Data2txt", "Summary")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "font.family": "DejaVu Sans", "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    return plt


def save(fig, out_dir: str | Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=200)
    return paths


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

def _fmt(m: dict) -> str:
    return f"{m['mean']:.3f}" if m.get("sd") is None else f"{m['mean']:.3f} ± {m['sd']:.3f}"


def main_table(final: dict, part: str = "full") -> pd.DataFrame:
    """Table 1: every system on the final test at its frozen dev thresholds, sorted by span F1."""
    rows = []
    for unit, r in final["systems"][part].items():
        o, ci = r["scores"]["tuned"]["overall"], r["ci"]
        rows.append({
            "System": NAMES.get(unit, unit), "Kind": r["kind"], "Seeds": len(r["runs"]),
            "Example F1": _fmt(o["example_f1"]),
            "Ex. F1 95% CI": f"[{ci['example_f1']['low']:.3f}, {ci['example_f1']['high']:.3f}]",
            "Span F1": _fmt(o["span_f1"]),
            "Span F1 95% CI": f"[{ci['span_f1']['low']:.3f}, {ci['span_f1']['high']:.3f}]",
            "Word F1": _fmt(o["word_f1"]), "Ex. AUROC": _fmt(o["example_auroc"]),
            "Word AUROC": _fmt(o["word_auroc"]), "_sort": o["span_f1"]["mean"]})
    return pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)


def task_table(final: dict, part: str = "full") -> pd.DataFrame:
    """Table 2: example and span F1 per task (frozen thresholds; our runs as seed means)."""
    rows = []
    for unit, r in final["systems"][part].items():
        row = {"System": NAMES.get(unit, unit), "_sort": r["scores"]["tuned"]["overall"]["span_f1"]["mean"]}
        for task in TASKS:
            m = r["scores"]["tuned"][task]
            row[f"{task} ex. F1"] = round(m["example_f1"]["mean"], 3)
            row[f"{task} span F1"] = round(m["span_f1"]["mean"], 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)


def comparisons_table(final: dict, part: str = "full") -> pd.DataFrame:
    rows = []
    for qid, c in final["comparisons"][part].items():
        for metric in METRIC_NAMES:
            if metric in c:
                rows.append({"Q": qid, "Comparison": c["label"], "Metric": METRIC_NAMES[metric],
                             "Difference": round(c[metric]["diff"], 3), "CI low": round(c[metric]["low"], 3),
                             "CI high": round(c[metric]["high"], 3),
                             "Role": "primary" if qid == "Q1" else "secondary"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def _legend_kinds(ax, kinds):
    from matplotlib.lines import Line2D

    labels = {"ours": "Our runs (R0–R4)", "existing": "Existing Turkish detectors", "classical": "Classical baselines"}
    handles = [Line2D([], [], marker="o", ls="", color=KIND_COLOR[k], markersize=5, label=labels[k]) for k in kinds]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)


def main_results(final: dict, part: str = "full"):
    """Figure 1: example and span F1 with 95% CIs for every system, sorted by span F1."""
    plt = _plt()
    units = sorted(final["systems"][part].items(), key=lambda kv: kv[1]["scores"]["tuned"]["overall"]["span_f1"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
    ys = range(len(units))
    for ax, metric in zip(axes, METRIC_NAMES):
        for y, (unit, r) in zip(ys, units):
            ci, color = r["ci"][metric], KIND_COLOR[r["kind"]]
            ax.plot([ci["low"], ci["high"]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(ci["point"], y, "o", color=color, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel(METRIC_NAMES[metric] + " (final test, 95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ys), [NAMES.get(u, u) for u, _ in units])
    for label, (unit, _) in zip(axes[0].get_yticklabels(), units):
        if unit == final["decisions"]["final_model"]:
            label.set_fontweight("bold")
    _legend_kinds(fig.add_axes([0.3, 0.98, 0.4, 0.01], frameon=False, xticks=[], yticks=[]),
                  [k for k in ("ours", "existing", "classical") if any(r["kind"] == k for _, r in units)])
    fig.tight_layout()
    return fig


def forest(final: dict, part: str = "full"):
    """Figure 2: Q1–Q5 differences (a − b) with 95% CIs. Only Q1 is confirmatory."""
    plt = _plt()
    comps = list(final["comparisons"][part].items())[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for ax, metric in zip(axes, METRIC_NAMES):
        for y, (qid, c) in enumerate(comps):
            if metric not in c:
                continue
            d = c[metric]
            color = SLOTS[0] if qid == "Q1" else INK_2
            ax.plot([d["low"], d["high"]], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(d["diff"], y, "D" if qid == "Q1" else "o", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(0, color=INK_MUTED, lw=1)
        ax.set_xlabel(f"Difference in {METRIC_NAMES[metric]} (a − b, 95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(comps)), [f"{q}: {c['label']}" for q, c in comps])
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    fig.tight_layout()
    return fig


def per_task(final: dict, units: tuple[str, ...] | None = None, part: str = "full"):
    """Figure 3: span F1 per task for our final runs and the strongest existing detectors."""
    plt = _plt()
    systems = final["systems"][part]
    if units is None:
        existing = sorted((u for u, r in systems.items() if r["kind"] == "existing"),
                          key=lambda u: -systems[u]["scores"]["tuned"]["overall"]["span_f1"]["mean"])[:2]
        units = tuple(dict.fromkeys([final["decisions"]["final_model"], "r4", *existing,
                                     final["decisions"]["best_baseline"]]))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    width = 0.8 / len(units)
    for i, unit in enumerate(units):
        vals = [systems[unit]["scores"]["tuned"][t]["span_f1"]["mean"] for t in TASKS]
        xs = [t + (i - (len(units) - 1) / 2) * width for t in range(len(TASKS))]
        ax.bar(xs, vals, width * 0.9, color=SLOTS[i], label=NAMES.get(unit, unit))
    ax.set_xticks(range(len(TASKS)), TASKS)
    ax.set_ylabel("Span F1 (final test)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6)
    fig.tight_layout()
    return fig


def threshold_curves(frozen: dict, groups: dict[str, list[str]] | None = None):
    """Figure 4: dev example and span F1 against τ; markers at each system's frozen τ."""
    plt = _plt()
    curves, systems = frozen["threshold_curves"], frozen["systems"]
    if groups is None:
        ours = lambda g: sorted(n for n, s in systems.items() if s.get("group") == g)
        groups = {NAMES["r2"]: ours("r2"), NAMES["r0"]: ours("r0"),
                  NAMES["newmind_turkembed4sts"]: ["newmind_turkembed4sts"],
                  NAMES["newmind_modernbert_tr"]: ["newmind_modernbert_tr"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    taus = sorted(float(t) for t in next(iter(curves.values())))
    for ax, metric, tau_key in zip(axes, METRIC_NAMES, ("tau_example", "tau_span")):
        for i, (label, names) in enumerate(groups.items()):
            ys = [sum(curves[n][f"{t:g}"][metric] for n in names) / len(names) for t in taus]
            ax.plot(taus, ys, color=SLOTS[i], lw=2, label=label)
            chosen = [systems[n][tau_key] for n in names]
            for c in sorted(set(chosen)):
                y = sum(curves[n][f"{c:g}"][metric] for n in names) / len(names)
                ax.plot(c, y, "o", color=SLOTS[i], markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
        ax.set_xlabel("Threshold τ (dev)")
        ax.set_title(METRIC_NAMES[metric] + (" (seed mean for our runs)" if metric == "example_f1" else ""),
                     loc="left", color=INK_2)
    axes[0].set_ylabel("F1 on dev")
    axes[1].legend(frameon=False, loc="lower center", fontsize=6)
    fig.tight_layout()
    return fig


def training_curves(train: dict, groups=("r1", "r2", "r3", "r4"), metric: str = "eval_word_f1"):
    """Figure 5: validation word F1 per half epoch, one panel per run group, one line per seed."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(groups), figsize=(7.16, 2.0), sharey=True)
    for ax, group in zip(axes, groups):
        runs = sorted((r for r in train["runs"] if r["group"] == group), key=lambda r: r["seed"])
        for i, r in enumerate(runs):
            hist = [h for h in r["validation_history"] if metric in h]
            if hist:
                ax.plot([h["epoch"] for h in hist], [h[metric] for h in hist], color=SLOTS[i], lw=2,
                        marker="o", markersize=3, label=f"seed {r['seed']}")
        ax.set_title(NAMES.get(group, group), loc="left", color=INK_2)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation word F1")
    axes[-1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    return fig


def build_all(train: dict, frozen: dict, final: dict, out_dir: str | Path) -> dict:
    """Write every planned figure and table; returns the written paths."""
    out_dir = Path(out_dir)
    written = {}
    for name, fig in (("fig1_main_results", main_results(final)), ("fig2_q1_q5_forest", forest(final)),
                      ("fig3_span_f1_per_task", per_task(final)), ("fig4_threshold_curves", threshold_curves(frozen)),
                      ("fig5_training_curves", training_curves(train))):
        written[name] = [str(p) for p in save(fig, out_dir, name)]
        fig.clf()
    for name, table in (("table1_main", main_table(final)), ("table1_main_clean", main_table(final, "clean")),
                        ("table2_per_task", task_table(final)), ("table3_q1_q5", comparisons_table(final))):
        csv, tex = out_dir / f"{name}.csv", out_dir / f"{name}.tex"
        table.to_csv(csv, index=False, encoding="utf-8")
        tex.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")
        written[name] = [str(csv), str(tex)]
    return written
