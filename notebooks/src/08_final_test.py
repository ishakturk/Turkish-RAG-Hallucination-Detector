# colab-runtime: cpu
# %% [markdown]
# # 08 · Final test: scored once
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# This notebook opens the sealed final test (2,160 responses, 360 sources) and scores every system
# **once**, exactly as fixed in `docs/analysis-plan.md` (§3–5):
# - thresholds come from the **committed** `results/07_thresholds.json` (tuned on dev, frozen before this);
# - example metrics at τ_example, word and span metrics at τ_span, plus everything at τ = 0.5;
# - our runs as mean ± sd over seeds; 95% CIs from 2,000 bootstrap replicates over **sources**, seed-averaged;
# - **Q1** (primary): final model − best baseline, with paired permutation tests per seed (10,000 permutations);
# - **Q2–Q5** (secondary): effect sizes with CIs, no multiple-comparison correction;
# - the same for the clean final test (2,146 responses), reported next to the full one.
#
# **Run it once.** If `08_final_test.json` already exists on Drive, the notebook stops. A rerun is only
# allowed with a written reason (`RERUN_REASON`), which is saved in the output and goes into the
# analysis plan's amendments (§10).
#
# **Runtime:** CPU, ~10–15 min. **Secrets:** `GH_TOKEN`.
# **Output:** `MyDrive/hdtr/results/08_final_test.json`. Send it back.

# %%
RERUN_REASON = None   # leave None. Only set (with a sentence) if Claude asks for a documented rerun.
N_BOOT = 2000
N_PERM = 10000

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)

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
import json, time
import pandas as pd

from hdtr import data as D
from hdtr.final import (COMPARISONS, OUR_GROUPS, compare, file_sha256, load_probs, mean_sd,
                        permutation_per_seed, q1_verdict, records_at, run_scores, subset, system_cis)

OUT_PATH = f"{DRIVE_DIR}/results/08_final_test.json"
if os.path.exists(OUT_PATH) and not RERUN_REASON:
    raise RuntimeError("The final test has been scored already (08_final_test.json exists). "
                       "It is scored once; ask Claude before rerunning.")

# %% [markdown]
# ## Frozen thresholds and decisions
# Read from the repository (the committed copy), and every predictions file is checked against the
# fingerprint recorded in 07. A changed file stops the notebook.

# %%
thr_path = f"{REPO_DIR}/results/07_thresholds.json"
if not os.path.exists(thr_path):
    raise RuntimeError("results/07_thresholds.json is not committed yet: send the 07 output to Claude first.")
with open(thr_path, encoding="utf-8") as f:
    frozen = json.load(f)
systems, decisions = frozen["systems"], frozen["decisions"]
print("thresholds frozen at", frozen["time"], "| commit", frozen["commit"])
print("final model:", decisions["final_model"], "| R4 base:", decisions["r4_base"],
      "| best baseline:", decisions["best_baseline"])

changed = [n for n, s in systems.items() if file_sha256(f"{DRIVE_DIR}/predictions/{s['file']}") != s["sha256"]]
if changed:
    raise RuntimeError(f"Predictions changed since thresholds were frozen: {changed}")
print(f"all {len(systems)} prediction files match their 07 fingerprints")

# %% [markdown]
# ## Data: the final test
# `test` = official RAGTruth-TR test split minus dev. `test_clean` drops the audit's hard flags.

# %%
tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
assert signals["pass"].mean() > 0.99, "alignment changed: rerun notebook 01"
tr["source_id"] = signals["source_id"].values
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
test = tr.loc[splits["test"]]
clean_ids = set(splits["test_clean"])
keep_clean = [i in clean_ids for i in test.index]
print("final test:", len(test), "responses,", test["source_id"].nunique(), "sources | clean:", sum(keep_clean))
print(test.groupby("task_type").size().to_dict())

# %% [markdown]
# ## Score every system
# For each run: records at τ = 0.5, τ_example and τ_span (frozen), then metrics on the full and clean test.

# %%
records = {}       # system name -> {"full": records_by_tau, "clean": records_by_tau}
scores = {}        # system name -> {"full": run_scores, "clean": run_scores}
for name, s in systems.items():
    start = time.perf_counter()
    probs = load_probs(f"{DRIVE_DIR}/predictions/{s['file']}", list(test.index))
    full = records_at(test, probs, s)
    records[name] = {"full": full, "clean": subset(full, keep_clean)}
    scores[name] = {k: run_scores(v) for k, v in records[name].items()}
    o = scores[name]["full"]["tuned"]["overall"]
    print(f"{name:30s} ex-F1 {o['example_f1']:.3f} span-F1 {o['span_f1']:.3f} "
          f"ex-AUROC {o['example_auroc']:.3f} ({time.perf_counter() - start:.0f}s)")

# %% [markdown]
# ## Systems as units
# An existing or classical detector is one run; each of our groups (R0–R4) is its seeds.

# %%
units = {}
for name, s in systems.items():
    key = s["group"] if s["kind"] == "ours" else name
    units.setdefault(key, {"kind": s["kind"], "runs": []})["runs"].append(name)
alias = {"final": decisions["final_model"], "best_baseline": decisions["best_baseline"], "r4_base": decisions["r4_base"]}
print({k: len(v["runs"]) for k, v in units.items()})

def unit_runs(unit, part):
    return [records[n][part] for n in units[unit]["runs"]]

# %% [markdown]
# ## Main table with 95% CIs (full and clean final test)

# %%
report = {"full": {}, "clean": {}}
for part in report:
    for unit, u in units.items():
        report[part][unit] = {"kind": u["kind"], "runs": u["runs"],
                              "scores": mean_sd([scores[n][part] for n in u["runs"]]),
                              "ci": system_cis(unit_runs(unit, part), n_boot=N_BOOT)}

def main_table(part, tau="tuned", key="overall"):
    rows = []
    for unit, r in report[part].items():
        m, ci = r["scores"][tau][key], r["ci"]
        row = {"system": unit, "kind": r["kind"], "seeds": len(r["runs"])}
        for name in ("example_f1", "span_f1", "word_f1", "example_auroc", "word_auroc"):
            if name in m:
                row[name] = m[name]["mean"]
                row[name + " sd"] = m[name]["sd"]
        if key == "overall" and tau == "tuned":
            for name in ("example_f1", "span_f1"):
                row[name + " CI"] = f"[{ci[name]['low']:.3f}, {ci[name]['high']:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("system").sort_values("span_f1", ascending=False)

for part in ("full", "clean"):
    print(f"\n{part} final test, frozen thresholds")
    display(main_table(part).round(3))
print("\nfull final test, τ = 0.5")
display(main_table("full", tau="tau_0.5").round(3))

# %% [markdown]
# ## Per task (full final test, frozen thresholds)

# %%
per_task = []
for unit, r in report["full"].items():
    for task in ("QA", "Data2txt", "Summary"):
        m = r["scores"]["tuned"][task]
        per_task.append({"system": unit, "task": task, "ex_f1": m["example_f1"]["mean"],
                         "span_f1": m["span_f1"]["mean"], "ex_auroc": m["example_auroc"]["mean"]})
display(pd.DataFrame(per_task).pivot(index="system", columns="task").round(3))

# %% [markdown]
# ## Q1–Q5: differences with 95% CIs
# Seed-averaged difference a − b; resampling sources (the same resample for both systems).
# Only Q1 supports a "better than" claim (analysis plan §1).

# %%
comparisons = {"full": {}, "clean": {}}
for part in comparisons:
    for qid, label, a, b in COMPARISONS:
        a, b = alias.get(a, a), alias.get(b, b)
        if a not in units or b not in units:
            comparisons[part][qid] = {"label": label, "a": a, "b": b, "error": "system missing"}
            continue
        comparisons[part][qid] = {"label": label, "a": a, "b": b,
                                  **compare(unit_runs(a, part), unit_runs(b, part), n_boot=N_BOOT)}

rows = []
for part, qs in comparisons.items():
    for qid, c in qs.items():
        for name in ("example_f1", "span_f1"):
            if name in c:
                rows.append({"test": part, "Q": qid, "comparison": f"{c['a']} − {c['b']}", "metric": name,
                             "diff": c[name]["diff"], "low": c[name]["low"], "high": c[name]["high"]})
display(pd.DataFrame(rows).set_index(["test", "Q", "metric"]).round(4))

# %% [markdown]
# ## Q1 decision
# Our final model is **better** than the best baseline on a primary metric if the 95% CI of the
# difference is entirely above 0. It is **robust** if each seed's permutation test gives p < 0.05.

# %%
final_unit, base_unit = alias["final"], alias["best_baseline"]
perms = permutation_per_seed(unit_runs(final_unit, "full"), unit_runs(base_unit, "full")[0], n_perm=N_PERM)
verdict = q1_verdict(comparisons["full"]["Q1"], perms)
for name, v in verdict.items():
    ps = [round(p["p_value"], 4) for p in perms[name]]
    print(f"{name}: diff {v['diff']:+.4f}, 95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] -> "
          f"{'BETTER' if v['better'] else 'not shown to be better'}; permutation p per seed {ps} "
          f"({'robust' if v['robust_all_seeds_p<0.05'] else 'not robust'})")

# %% [markdown]
# ## Save
# Send back `MyDrive/hdtr/results/08_final_test.json`.

# %%
results = {"notebook": "08_final_test", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "rerun_reason": RERUN_REASON, "thresholds_commit": frozen["commit"], "decisions": decisions,
           "n_boot": N_BOOT, "n_perm": N_PERM,
           "n": {"full": len(test), "clean": sum(keep_clean), "sources": int(test["source_id"].nunique())},
           "systems": report, "per_run": scores, "comparisons": comparisons,
           "q1": {"verdict": verdict, "permutations": perms}}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved", OUT_PATH)
