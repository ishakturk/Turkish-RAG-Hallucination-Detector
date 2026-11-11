# colab-runtime: cpu
# %% [markdown]
# # 07 · Freeze thresholds on dev
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Every system (the 7 existing detectors, the 2 classical baselines and all 13 of our runs) gets two
# thresholds, tuned on **dev only** (`docs/analysis-plan.md` §4.1):
# - **τ_example** maximises example F1, **τ_span** maximises span F1, over 0.05, 0.10, …, 0.95
#   (a tie goes to the value closest to 0.5).
#
# The notebook then re-applies the selection rules on these numbers (R4 base, final model, best
# baseline) and checks that they match the decisions already recorded. Each predictions file is
# fingerprinted (SHA-256), so notebook 08 can prove it scored exactly the predictions these
# thresholds were tuned on.
#
# **Nothing here touches the final test.** It only reads the saved predictions, so no GPU is needed.
#
# **Runtime:** CPU, ~10–15 min. **Secrets:** `GH_TOKEN`.
# **Output:** `MyDrive/hdtr/results/07_thresholds.json`. Send it back: I commit it, and notebook 08
# reads the committed copy. That commit is the proof that thresholds were frozen before the test.

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
from hdtr.final import CLASSICAL, EXISTING, apply_rules, file_sha256, load_probs, tune_system

# Decisions already recorded in docs/technical-roadmap.md from the 06 results. 07 recomputes them from
# the same predictions; a mismatch stops the notebook instead of silently changing the plan.
RECORDED = {"r4_base": "r2", "final_model": "r2", "best_baseline": None}  # best baseline is first decided here

# %% [markdown]
# ## Data: dev only
# Same RAGTruth-TR rows, source ids and splits as notebooks 02 and 06.

# %%
tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
assert signals["pass"].mean() > 0.99, "alignment changed: rerun notebook 01"
tr["source_id"] = signals["source_id"].values
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
dev = tr.loc[splits["dev"]]
print("dev:", len(dev), "responses,", dev["source_id"].nunique(), "sources")

# %% [markdown]
# ## Systems
# The list comes from the committed result files of notebooks 02 and 06, so no system can be added or
# left out by looking at scores.

# %%
with open(f"{REPO_DIR}/results/02_baselines_dev.json", encoding="utf-8") as f:
    baselines_02 = json.load(f)["systems"]
with open(f"{REPO_DIR}/results/06_train.json", encoding="utf-8") as f:
    runs_06 = json.load(f)["runs"]

systems = {}
for name, info in baselines_02.items():
    if "error" in info:
        print(f"{name}: failed in 02, reported as missing ({info['error'][:80]})")
        continue
    kind = "existing" if name in EXISTING else "classical" if name in CLASSICAL else "other"
    systems[name] = {"kind": kind, "group": None, "seed": None, "file": f"{name}.pkl"}
for r in sorted(runs_06, key=lambda r: r["name"]):
    systems[r["name"]] = {"kind": "ours", "group": r["group"], "seed": r["seed"], "file": f"06_{r['name']}.pkl"}

print(len(systems), "systems:", pd.Series([s["kind"] for s in systems.values()]).value_counts().to_dict())

# %% [markdown]
# ## Tune thresholds on dev

# %%
for name, s in systems.items():
    path = f"{DRIVE_DIR}/predictions/{s['file']}"
    start = time.perf_counter()
    probs = load_probs(path, list(dev.index))
    s.update(tune_system(dev, probs), sha256=file_sha256(path))
    o = s["dev"]["tuned"]["overall"]
    print(f"{name:30s} τ_ex {s['tau_example']:.2f} τ_span {s['tau_span']:.2f} | ex-F1 {o['example_f1']:.3f} "
          f"span-F1 {o['span_f1']:.3f} | selection {s['selection_score']:.3f} ({time.perf_counter() - start:.0f}s)")

# %% [markdown]
# ## Selection rules (analysis plan §4.3–4.5)
# - **R4 base:** best mean selection score among R1–R3 (gap < 0.01 → smaller model).
# - **Final model:** R4 if its score ≥ its base's, otherwise the base.
# - **Best baseline:** the existing detector with the highest selection score (classical ones are not candidates).

# %%
group_scores = {}
for s in systems.values():
    if s["kind"] == "ours":
        group_scores.setdefault(s["group"], []).append(s["selection_score"])
baseline_scores = {n: s["selection_score"] for n, s in systems.items() if s["kind"] == "existing"}
decisions = apply_rules(group_scores, baseline_scores)

print("group means:", {g: round(v, 4) for g, v in decisions["group_means"].items()})
print("existing detectors:", {n: round(v, 4) for n, v in sorted(baseline_scores.items(), key=lambda kv: -kv[1])})
print("R4 base:", decisions["r4_base"], "| final model:", decisions["final_model"],
      "| best baseline:", decisions["best_baseline"])
for key, expected in RECORDED.items():
    if expected is not None and decisions[key] != expected:
        raise RuntimeError(f"{key} is {decisions[key]!r} here but {expected!r} was recorded from 06. "
                           "Stop and send this output to Claude before going further.")

# %% [markdown]
# ## Dev table at the frozen thresholds
# Our runs as mean ± sd over seeds. Example metrics use τ_example; word and span metrics use τ_span.

# %%
rows = []
for name, s in systems.items():
    o = s["dev"]["tuned"]["overall"]
    rows.append({"system": s["group"] or name, "kind": s["kind"], "tau_ex": s["tau_example"], "tau_span": s["tau_span"],
                 "ex_f1": o["example_f1"], "span_f1": o["span_f1"], "word_f1": o["word_f1"],
                 "ex_auroc": o["example_auroc"], "word_auroc": o["word_auroc"], "selection": s["selection_score"],
                 **{f"span_f1 {t}": s["dev"]["tuned"][t]["span_f1"] for t in ("QA", "Data2txt", "Summary")}})
table = pd.DataFrame(rows)
display(table.groupby(["kind", "system"]).agg(["mean", "std"]).round(3)
        .sort_values(("selection", "mean"), ascending=False))

# %% [markdown]
# ## Save
# Send back `MyDrive/hdtr/results/07_thresholds.json`.

# %%
frozen = {"notebook": "07_thresholds", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
          "rule": "docs/analysis-plan.md §4", "decisions": decisions,
          "systems": {name: {k: v for k, v in s.items() if k != "curve"} for name, s in systems.items()},
          "threshold_curves": {name: s["curve"] for name, s in systems.items()}}
out_path = f"{DRIVE_DIR}/results/07_thresholds.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(frozen, f, ensure_ascii=False, indent=2)
print("saved", out_path)
