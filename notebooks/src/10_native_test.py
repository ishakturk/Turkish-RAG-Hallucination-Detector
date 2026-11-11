# %% [markdown]
# # 10 · Native Turkish test set: scored once
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Q6 of `docs/analysis-plan.md`: how do the detectors do on natural Turkish answers instead of
# machine-translated ones? The native set has 184 answers written in Turkish by Qwen3-8B and DeepSeek
# for questions about Turkish Wikipedia and Turkish law (notebooks 05a/05b).
# - **Labels:** example-level reference labels from two LLM label sets, with disagreements adjudicated by
#   the user, validated by a human on a blind 50-row sample (§10, amendments 1–2: κ 0.38, 86% agreement).
#   Gold spans are Claude-proposed (`data/native_spans.json`).
# - **Thresholds:** each system's **frozen dev thresholds** from `results/07_thresholds.json` (no tuning here).
# - **Primary:** example F1 and example AUROC. **Secondary:** span F1. Q6 is **descriptive**; no comparison
#   on this set supports a "better than" claim.
# - **Uncertainty:** 2,000 bootstrap replicates over **contexts**; our runs are seed-averaged.
#
# **Run it once.** If `10_native_test.json` already exists on Drive, the notebook stops.
#
# **Runtime:** L4 GPU, ~20–30 min (22 systems × 184 answers, mostly model loading). **Secrets:** `GH_TOKEN`.
# **Needs on Drive:** `native/native_test.parquet` (05b), `models/` (06), `predictions/` (02, 06).
# **Output:** `MyDrive/hdtr/results/10_native_test.json`. Send it back.

# %%
RERUN_REASON = None   # leave None. Only set (with a sentence) if Claude asks for a documented rerun.
N_BOOT = 2000
N_CHECK = 20          # dev rows re-predicted per system to check that each model loads as it was scored

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "predictions"):
    os.makedirs(f"{DRIVE_DIR}/{sub}", exist_ok=True)

token = userdata.get("GH_TOKEN")
if not token:
    raise RuntimeError("GH_TOKEN secret is missing or 'Notebook access' is off (key icon in the left bar).")

def git(*args):
    """Run git with the token passed through the environment, never in the command line,
    so it cannot appear in error messages or in .git/config."""
    header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
           "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
           "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {header}"}
    r = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip().replace(token, '***')}\n"
                           "If it says 'not found' or 'authentication', check that the token can read this repo "
                           "(Repository access + Contents: Read-only).")
    return r.stdout.strip()

if not os.path.exists(REPO_DIR):
    git("clone", "-q", "--depth", "1", f"https://github.com/{REPO}.git", REPO_DIR)
else:
    git("-C", REPO_DIR, "pull", "-q", "origin", "main")
commit = git("-C", REPO_DIR, "rev-parse", "--short", "HEAD")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[colab]"], check=True)
sys.path.insert(0, f"{REPO_DIR}/src")
print("repo commit:", commit)

# %%
import gc, json, pickle, time
import numpy as np
import pandas as pd
import torch

from hdtr import data as D
from hdtr import native_eval as NE
from hdtr.baselines import LexicalOverlap, WordLogReg
from hdtr.final import COMPARISONS, compare, file_sha256, load_probs, mean_sd, records_at, run_scores, subset, system_cis
from hdtr.infer import load_detector, predict_many

OUT_PATH = f"{DRIVE_DIR}/results/10_native_test.json"
if os.path.exists(OUT_PATH) and not RERUN_REASON:
    raise RuntimeError("The native test set has been scored already (10_native_test.json exists). "
                       "It is scored once; ask Claude before rerunning.")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (this will be slow)")

# %% [markdown]
# ## Frozen thresholds and the labeled native set
# Thresholds and decisions come from the committed 07 file. Reference labels, spans and the labeled
# answers come from the repository; every answer in `native_test.parquet` must equal the answer that
# was labeled.

# %%
with open(f"{REPO_DIR}/results/07_thresholds.json", encoding="utf-8") as f:
    frozen = json.load(f)
systems, decisions = frozen["systems"], frozen["decisions"]
print("thresholds frozen at", frozen["time"], "| final model:", decisions["final_model"],
      "| best baseline:", decisions["best_baseline"])

reference = pd.read_csv(f"{REPO_DIR}/data/native_reference_labels.csv")
with open(f"{REPO_DIR}/data/native_spans.json", encoding="utf-8") as f:
    spans = json.load(f)["spans"]
sheet = pd.read_csv(f"{REPO_DIR}/data/native_labeling_blank.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
native = NE.native_frame(pd.read_parquet(f"{DRIVE_DIR}/native/native_test.parquet"), reference, spans,
                         dict(zip(sheet["id"], sheet["answer"])))
print(len(native), "answers |", native["reference_label"].sum(), "hallucinated |",
      native["source_id"].nunique(), "contexts")
display(native.groupby(["domain", "generator"])["reference_label"].agg(["sum", "count"]))
fingerprints = {name: file_sha256(f"{REPO_DIR}/data/{name}") for name in
                ("native_reference_labels.csv", "native_spans.json", "native_labeling_blank.csv")}

# %% [markdown]
# ## Dev rows for the load check
# For every neural system, `N_CHECK` dev rows are predicted again and compared with the predictions
# it was scored with in 07/08. A large difference means the model or weights did not load as before.

# %%
tr = D.load_ragtruth_tr()
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
check_rows = tr.loc[splits["dev"][:N_CHECK]]

def load_check(name, model, tokenizer, max_length, autocast):
    with autocast:
        probs, _ = predict_many(model, tokenizer, zip(check_rows["prompt"], check_rows["answer"]), max_length=max_length)
    saved = load_probs(f"{DRIVE_DIR}/predictions/{systems[name]['file']}", list(check_rows.index))
    return float(max(np.abs(p - s.astype(np.float32)).max() for p, s in zip(probs, saved) if len(p)))

# %% [markdown]
# ## Predict the native set
# Existing detectors from Hugging Face, our runs from `MyDrive/hdtr/models/`. Predictions are saved per
# system (`predictions/10_native_<system>.pkl`), so a disconnect only repeats the unfinished system.

# %%
pairs = list(zip(native["prompt"], native["answer"]))
timing, checks, failed = {}, {}, {}

def save_native(name, probs, info):
    with open(f"{DRIVE_DIR}/predictions/10_native_{name}.pkl", "wb") as f:
        pickle.dump({"row_ids": list(native.index), "probs": [p.astype(np.float16) for p in probs], **info}, f)

for name, s in systems.items():
    path = f"{DRIVE_DIR}/predictions/10_native_{name}.pkl"
    if os.path.exists(path) or s["kind"] == "classical":
        continue
    try:
        start = time.perf_counter()
        if s["kind"] == "existing":
            model_id, remote = NE.EXISTING_MODELS[name]
            tok, model = load_detector(model_id, **({"trust_remote_code": True} if remote else {}))
            max_length, autocast = 8192, torch.autocast("cuda", enabled=False)
        else:
            model_id = f"{DRIVE_DIR}/models/{name}"
            tok, model = load_detector(model_id)
            max_length, autocast = 4096, torch.autocast("cuda", dtype=torch.bfloat16)
        check = load_check(name, model, tok, max_length, autocast)
        with autocast:
            probs, t = predict_many(model, tok, pairs, max_length=max_length)
        save_native(name, probs, {"timing": t, "model_id": model_id, "load_check_max_abs_diff": check})
        print(f"{name:30s} load check max |Δp| {check:.4f}  {time.perf_counter() - start:.0f}s")
        del model
    except Exception as exc:   # one broken model must not stop the others
        failed[name] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"{name}: FAILED — {failed[name]}")
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Classical baselines
# Refit exactly as in notebook 02 (same samples, same seeds) and checked against their saved dev predictions.

# %%
if any(s["kind"] == "classical" and not os.path.exists(f"{DRIVE_DIR}/predictions/10_native_{n}.pkl")
       for n, s in systems.items()):
    train = tr.loc[splits["train_clean"]]
    fit_sample = train.sample(4000, random_state=0)
    lex = LexicalOverlap().fit(train["prompt"].sample(3000, random_state=0), train["answer"].sample(3000, random_state=0))
    logreg = WordLogReg(lex).fit(fit_sample["prompt"], fit_sample["answer"], fit_sample["labels"])
    for name, model in {"classical_lexical_overlap": lex, "classical_word_logreg": logreg}.items():
        saved = load_probs(f"{DRIVE_DIR}/predictions/{systems[name]['file']}", list(check_rows.index))
        again = [model.predict_char_probs(p, a) for p, a in zip(check_rows["prompt"], check_rows["answer"])]
        check = float(max(np.abs(p - s.astype(np.float32)).max() for p, s in zip(again, saved) if len(p)))
        probs = [model.predict_char_probs(p, a) for p, a in pairs]
        save_native(name, probs, {"load_check_max_abs_diff": check})
        print(f"{name:30s} refit check max |Δp| {check:.4f}")

# %%
probs_by_system = {}
for name in systems:
    path = f"{DRIVE_DIR}/predictions/10_native_{name}.pkl"
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        saved = pickle.load(f)
    assert saved["row_ids"] == list(native.index)
    probs_by_system[name] = [p.astype(np.float32) for p in saved["probs"]]
    checks[name] = saved.get("load_check_max_abs_diff")
    timing[name] = saved.get("timing")
bad = {n: c for n, c in checks.items() if c is not None and c > 0.05}
print(f"{len(probs_by_system)}/{len(systems)} systems predicted | failed: {list(failed) or 'none'}")
print("load checks above 0.05 (send these to Claude):", bad or "none")

# %% [markdown]
# ## Score with frozen thresholds
# Example metrics at τ_example, span and word metrics at τ_span, plus τ = 0.5. "Tasks" here are the
# two domains (wikipedia, legal).

# %%
records, scores = {}, {}
for name, probs in probs_by_system.items():
    records[name] = records_at(native, probs, systems[name])
    scores[name] = run_scores(records[name])

units = {}
for name, s in systems.items():
    if name in records:
        key = s["group"] if s["kind"] == "ours" else name
        units.setdefault(key, {"kind": s["kind"], "runs": []})["runs"].append(name)
alias = {"final": decisions["final_model"], "best_baseline": decisions["best_baseline"], "r4_base": decisions["r4_base"]}

def unit_runs(unit, recs=records):
    return [recs[n] for n in units[unit]["runs"]]

report = {}
for unit, u in units.items():
    runs = unit_runs(unit)
    report[unit] = {"kind": u["kind"], "runs": u["runs"], "scores": mean_sd([scores[n] for n in u["runs"]]),
                    "ci": {**system_cis(runs, n_boot=N_BOOT),
                           "example_auroc": NE.example_auroc_ci([r["tau_0.5"] for r in runs], n_boot=N_BOOT)}}

rows = []
for unit, r in report.items():
    m, ci = r["scores"]["tuned"]["overall"], r["ci"]
    rows.append({"system": unit, "kind": r["kind"], "seeds": len(r["runs"]),
                 "ex_F1": m["example_f1"]["mean"], "ex_F1 CI": f"[{ci['example_f1']['low']:.3f}, {ci['example_f1']['high']:.3f}]",
                 "ex_AUROC": ci["example_auroc"]["point"],
                 "ex_AUROC CI": f"[{ci['example_auroc']['low']:.3f}, {ci['example_auroc']['high']:.3f}]",
                 "span_F1": m["span_f1"]["mean"], "span_F1 CI": f"[{ci['span_f1']['low']:.3f}, {ci['span_f1']['high']:.3f}]"})
display(pd.DataFrame(rows).set_index("system").sort_values("ex_F1", ascending=False).round(3))

# %% [markdown]
# ## Breakdowns (descriptive, means over seeds)
# By domain, by generator and by question type (original vs rewritten to be partly answerable).

# %%
breakdown = {}
for col in ("domain", "generator", "question_type"):
    for value in sorted(native[col].unique()):
        keep = list(native[col] == value)
        key = f"{col}={value}"
        breakdown[key] = {"n": sum(keep), "hallucinated": int(native.loc[keep, "reference_label"].sum()), "systems": {}}
        for unit, u in units.items():
            sub = [run_scores(subset(records[n], keep)) for n in u["runs"]]
            breakdown[key]["systems"][unit] = mean_sd(sub)["tuned"]["overall"]
rows = [{"subset": k, "system": unit, "ex_F1": m["example_f1"]["mean"], "span_F1": m["span_f1"]["mean"]}
        for k, b in breakdown.items() for unit, m in b["systems"].items()]
display(pd.DataFrame(rows).pivot(index="system", columns="subset", values="ex_F1").round(3))

# %% [markdown]
# ## Differences on the native set (descriptive)
# The same Q1–Q5 pairs as 08, plus the final model against every existing detector. Seed-averaged
# differences with 95% CIs over contexts. These describe the native set; they are not confirmatory.

# %%
pairs_to_compare = [(qid, label, alias.get(a, a), alias.get(b, b)) for qid, label, a, b in COMPARISONS]
pairs_to_compare += [(f"final_vs_{n}", f"final model − {n}", alias["final"], n)
                     for n, s in systems.items() if s["kind"] == "existing" and n != alias["best_baseline"]]
comparisons = {}
for qid, label, a, b in pairs_to_compare:
    if a not in units or b not in units:
        comparisons[qid] = {"label": label, "a": a, "b": b, "error": "system missing"}
        continue
    comparisons[qid] = {"label": label, "a": a, "b": b, **compare(unit_runs(a), unit_runs(b), n_boot=N_BOOT)}
rows = [{"id": q, "comparison": f"{c['a']} − {c['b']}", "metric": m, "diff": c[m]["diff"], "low": c[m]["low"], "high": c[m]["high"]}
        for q, c in comparisons.items() for m in ("example_f1", "span_f1") if m in c]
display(pd.DataFrame(rows).set_index(["id", "metric"]).round(4))

# %% [markdown]
# ## Translated vs native
# The same system on the final (translated) test from `results/08_final_test.json` and on the native set.

# %%
with open(f"{REPO_DIR}/results/08_final_test.json", encoding="utf-8") as f:
    final_test = json.load(f)
gap = {}
for unit in report:
    t = final_test["systems"]["full"].get(unit)
    if not t:
        continue
    tm, nm = t["scores"]["tuned"]["overall"], report[unit]["scores"]["tuned"]["overall"]
    gap[unit] = {m: {"translated": tm[m]["mean"], "native": nm[m]["mean"]} for m in ("example_f1", "span_f1", "example_auroc")}
display(pd.DataFrame({u: {f"{m} {k}": v for m, d in g.items() for k, v in d.items()} for u, g in gap.items()}).T.round(3))

# %% [markdown]
# ## Save
# Send back `MyDrive/hdtr/results/10_native_test.json`.

# %%
results = {"notebook": "10_native_test", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "rerun_reason": RERUN_REASON, "thresholds_commit": frozen["commit"], "decisions": decisions,
           "n_boot": N_BOOT, "fingerprints": fingerprints,
           "n": {"answers": len(native), "hallucinated": int(native["reference_label"].sum()),
                 "contexts": int(native["source_id"].nunique())},
           "failed": failed, "load_checks": checks, "timing": timing,
           "systems": report, "per_run": scores, "breakdown": breakdown, "comparisons": comparisons,
           "translated_vs_native": gap}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("saved", OUT_PATH)
