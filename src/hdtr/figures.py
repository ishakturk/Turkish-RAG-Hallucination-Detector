"""Report figures and tables (docs/analysis-plan.md §9) built from the committed result files.

Everything here reads results/06_train.json, results/07_thresholds.json, results/08_final_test.json and
results/10_native_test.json, so the figures can be rebuilt anywhere without predictions or a GPU. Figures
are sized for an IEEE two-column paper (3.5 in single column, 7.16 in double column) and saved as PDF (for
LaTeX) and PNG. ``lang="tr"`` gives Turkish labels for the Turkish report and slides.
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
NAMES_TR = {
    **NAMES, "r0": "R0 TabiBERT, Newmind tarifi", "r2": "R2 mmBERT-base (nihai)",
    "r3": "R3 KR Labs başlangıçlı", "r4": "R4 mmBERT + veri artırma",
    "newmind_turkembed4sts": "TurkEmbed4STS (en iyi referans)", "krlabs_v2_mmbert_zeroshot": "KR Labs v2 (sıfır atış)",
    "classical_lexical_overlap": "Sözcük örtüşmesi", "classical_word_logreg": "Kelime lojistik regresyon",
}
METRIC_NAMES = {"example_f1": "Example F1", "span_f1": "Span F1"}
TASKS = ("QA", "Data2txt", "Summary")
Q_LABELS_TR = {"Q1": "nihai model − en iyi referans", "Q2": "R1 (TabiBERT) − R2 (mmBERT-base)",
               "Q3": "R3 (KR Labs başlangıcı) − R2 (mmBERT-base)", "Q4": "R1 (bizim tarif) − R0 (Newmind tarifi)",
               "Q5": "R4 (veri artırmalı) − temel model"}
TEXT_TR = {
    "Example F1": "Örnek F1", "Span F1": "Span F1", "Example AUROC": "Örnek AUROC",
    "QA": "Soru-cevap", "Data2txt": "Veriden metin", "Summary": "Özetleme",
    " (final test, 95% CI)": " (nihai test, %95 GA)",
    "Our runs (R0–R4)": "Bizim modeller (R0–R4)", "Existing Turkish detectors": "Mevcut Türkçe dedektörler",
    "Classical baselines": "Klasik yöntemler", "Difference in {} (a − b, 95% CI)": "{} farkı (a − b, %95 GA)",
    "Span F1 (final test)": "Span F1 (nihai test)", "Threshold τ (dev)": "Eşik τ (geliştirme kümesi)",
    " (seed mean for our runs)": " (bizim modellerde tohum ortalaması)", "F1 on dev": "Geliştirme kümesinde F1",
    "seed {}": "tohum {}", "Epoch": "Devir", "Validation word F1": "Doğrulama kelime F1",
    "translated test": "çeviri test", "native test": "özgün Türkçe test",
    "Hallucinated answers (%)": "Halüsinasyonlu cevap (%)", "Generator": "Üreten model",
    "Question type": "Soru tipi", "original": "özgün", "partial": "kısmen cevaplanabilir",
    "qwen": "Qwen3-8B", "deepseek": "DeepSeek", "legal": "Hukuk", "wikipedia": "Vikipedi", "Domain": "Alan",
}
_LANG = "en"


def set_language(lang: str) -> None:
    global _LANG
    if lang not in ("en", "tr"):
        raise ValueError("lang must be 'en' or 'tr'")
    _LANG = lang


def t(text: str) -> str:
    return TEXT_TR.get(text, text) if _LANG == "tr" else text


def name(unit: str) -> str:
    return (NAMES_TR if _LANG == "tr" else NAMES).get(unit, unit)


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
            "System": name(unit), "Kind": r["kind"], "Seeds": len(r["runs"]),
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
        row = {"System": name(unit), "_sort": r["scores"]["tuned"]["overall"]["span_f1"]["mean"]}
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


def native_table(native: dict) -> pd.DataFrame:
    """Table 4: every system on the native test set (frozen dev thresholds), sorted by example F1."""
    rows = []
    for unit, r in native["systems"].items():
        o, ci = r["scores"]["tuned"]["overall"], r["ci"]
        rows.append({
            "System": name(unit), "Kind": r["kind"], "Seeds": len(r["runs"]),
            "Example F1": _fmt(o["example_f1"]),
            "Ex. F1 95% CI": f"[{ci['example_f1']['low']:.3f}, {ci['example_f1']['high']:.3f}]",
            "Precision": f"{o['example_p']['mean']:.2f}", "Recall": f"{o['example_r']['mean']:.2f}",
            "Ex. AUROC": f"{ci['example_auroc']['point']:.3f}",
            "AUROC 95% CI": f"[{ci['example_auroc']['low']:.3f}, {ci['example_auroc']['high']:.3f}]",
            "Span F1": _fmt(o["span_f1"]), "_sort": o["example_f1"]["mean"]})
    return pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def _legend_kinds(ax, kinds):
    from matplotlib.lines import Line2D

    labels = {"ours": "Our runs (R0–R4)", "existing": "Existing Turkish detectors", "classical": "Classical baselines"}
    handles = [Line2D([], [], marker="o", ls="", color=KIND_COLOR[k], markersize=5, label=t(labels[k])) for k in kinds]
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
        ax.set_xlabel(t(METRIC_NAMES[metric]) + t(" (final test, 95% CI)"))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ys), [name(u) for u, _ in units])
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
        ax.set_xlabel(t("Difference in {} (a − b, 95% CI)").format(t(METRIC_NAMES[metric])))
        ax.grid(axis="y", visible=False)
    labels = [f"{q}: {Q_LABELS_TR.get(q, c['label']) if _LANG == 'tr' else c['label']}" for q, c in comps]
    axes[0].set_yticks(range(len(comps)), labels)
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
        vals = [systems[unit]["scores"]["tuned"][task]["span_f1"]["mean"] for task in TASKS]
        xs = [k + (i - (len(units) - 1) / 2) * width for k in range(len(TASKS))]
        ax.bar(xs, vals, width * 0.9, color=SLOTS[i], label=name(unit))
    ax.set_xticks(range(len(TASKS)), [t(task) for task in TASKS])
    ax.set_ylabel(t("Span F1 (final test)"))
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
        groups = {name("r2"): ours("r2"), name("r0"): ours("r0"),
                  name("newmind_turkembed4sts"): ["newmind_turkembed4sts"],
                  name("newmind_modernbert_tr"): ["newmind_modernbert_tr"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    taus = sorted(float(x) for x in next(iter(curves.values())))
    for ax, metric, tau_key in zip(axes, METRIC_NAMES, ("tau_example", "tau_span")):
        for i, (label, names) in enumerate(groups.items()):
            ys = [sum(curves[n][f"{x:g}"][metric] for n in names) / len(names) for x in taus]
            ax.plot(taus, ys, color=SLOTS[i], lw=2, label=label)
            chosen = [systems[n][tau_key] for n in names]
            for c in sorted(set(chosen)):
                y = sum(curves[n][f"{c:g}"][metric] for n in names) / len(names)
                ax.plot(c, y, "o", color=SLOTS[i], markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
        ax.set_xlabel(t("Threshold τ (dev)"))
        ax.set_title(t(METRIC_NAMES[metric]) + (t(" (seed mean for our runs)") if metric == "example_f1" else ""),
                     loc="left", color=INK_2)
    axes[0].set_ylabel(t("F1 on dev"))
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
                        marker="o", markersize=3, label=t("seed {}").format(r["seed"]))
        ax.set_title(name(group), loc="left", color=INK_2)
        ax.set_xlabel(t("Epoch"))
    axes[0].set_ylabel(t("Validation word F1"))
    axes[-1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    return fig


def translated_vs_native(native: dict, metrics=("example_f1", "example_auroc")):
    """Figure 6: each system on the translated final test (open marker) and the native set (filled),
    sorted by native example F1. Colour = kind."""
    plt = _plt()
    gap, systems = native["translated_vs_native"], native["systems"]
    units = sorted(gap, key=lambda u: gap[u]["example_f1"]["native"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.16, 3.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        for y, unit in enumerate(units):
            a, b, color = gap[unit][metric]["translated"], gap[unit][metric]["native"], KIND_COLOR[systems[unit]["kind"]]
            ax.plot([a, b], [y, y], color=color, lw=1.5, alpha=0.5, solid_capstyle="round")
            ax.plot(a, y, "o", color="white", markeredgecolor=color, markeredgewidth=1.5, markersize=5)
            ax.plot(b, y, "o", color=color, markeredgecolor="white", markeredgewidth=0.8, markersize=6)
        ax.set_xlabel(t({"example_f1": "Example F1", "example_auroc": "Example AUROC", "span_f1": "Span F1"}[metric]))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(units)), [name(u) for u in units])
    for label, unit in zip(axes[0].get_yticklabels(), units):
        if unit in (native["decisions"]["final_model"], "r4"):
            label.set_fontweight("bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="white", markeredgecolor=INK_2, markeredgewidth=1.5,
                      markersize=5, label=t("translated test")),
               Line2D([], [], marker="o", ls="", color=INK_2, markersize=6, label=t("native test"))]
    handles += [Line2D([], [], marker="o", ls="", color=KIND_COLOR[k], markersize=5, label=t(lab)) for k, lab in
                (("ours", "Our runs (R0–R4)"), ("existing", "Existing Turkish detectors"), ("classical", "Classical baselines"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    return fig


def native_rates(native: dict):
    """Figure 7: share of hallucinated answers in the native set by generator, question type and domain."""
    plt = _plt()
    groups = (("Generator", "generator", ("qwen", "deepseek")), ("Question type", "question_type", ("partial", "original")),
              ("Domain", "domain", ("wikipedia", "legal")))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    for ax, (title, col, values) in zip(axes, groups):
        rates, labels = [], []
        for v in values:
            b = native["breakdown"][f"{col}={v}"]
            rates.append(100 * b["hallucinated"] / b["n"])
            labels.append(f"{t(v)}\n{b['hallucinated']}/{b['n']}")
        bars = ax.bar(range(len(values)), rates, 0.6, color=SLOTS[0])
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(values)), labels)
        ax.set_title(t(title), loc="left", color=INK_2)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 60)
    axes[0].set_ylabel(t("Hallucinated answers (%)"))
    fig.tight_layout()
    return fig


def build_all(train: dict, frozen: dict, final: dict, out_dir: str | Path, native: dict | None = None,
              lang: str = "en") -> dict:
    """Write every planned figure and table; returns the written paths. Native figures and table need
    ``native`` (results/10_native_test.json)."""
    set_language(lang)
    try:
        out_dir = Path(out_dir)
        written = {}
        figs = [("fig1_main_results", lambda: main_results(final)), ("fig2_q1_q5_forest", lambda: forest(final)),
                ("fig3_span_f1_per_task", lambda: per_task(final)),
                ("fig4_threshold_curves", lambda: threshold_curves(frozen)),
                ("fig5_training_curves", lambda: training_curves(train))]
        tables = [("table1_main", main_table(final)), ("table1_main_clean", main_table(final, "clean")),
                  ("table2_per_task", task_table(final)), ("table3_q1_q5", comparisons_table(final))]
        if native is not None:
            figs += [("fig6_translated_vs_native", lambda: translated_vs_native(native)),
                     ("fig7_native_rates", lambda: native_rates(native))]
            tables.append(("table4_native", native_table(native)))
        for fig_name, make in figs:
            fig = make()
            written[fig_name] = [str(p) for p in save(fig, out_dir, fig_name)]
            fig.clf()
        for table_name, table in tables:
            csv, tex = out_dir / f"{table_name}.csv", out_dir / f"{table_name}.tex"
            table.to_csv(csv, index=False, encoding="utf-8")
            tex.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")
            written[table_name] = [str(csv), str(tex)]
        return written
    finally:
        set_language("en")
