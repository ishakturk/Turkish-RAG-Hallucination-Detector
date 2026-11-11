# colab-runtime: cpu
# %% [markdown]
# # 09 · Figures, tables and exploratory comparisons
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# 1. **Report figures and tables** (analysis plan §9), built only from the committed result files
#    (`results/06_train.json`, `07_thresholds.json`, `08_final_test.json`). Nothing is re-scored.
# 2. **Exploratory comparisons (not pre-registered):** the final model (R2) against *every* existing
#    detector on the final test, with the same seed-averaged paired bootstrap as Q1. The pre-registered
#    Q1 compares R2 with TurkEmbed4STS-HD, the best baseline on dev; on the test, ModernBERT-tr and
#    bge-m3 score higher than TurkEmbed4STS. These intervals show how R2 compares with them. They are
#    reported as exploratory, with no correction for the 7 comparisons, and never replace Q1.
#
# The native test set (Q6) gets its own figure once it is labeled and scored.
#
# **Runtime:** CPU, ~15 min (the exploratory part reads the test predictions). **Secrets:** `GH_TOKEN`.
# **Output:** `MyDrive/hdtr/figures/` (PDF + PNG figures, CSV + LaTeX tables) and
# `MyDrive/hdtr/results/09_exploratory.json` (send this one back).

# %%
N_BOOT = 2000

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "figures"):
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
import json, time
import pandas as pd
from IPython.display import Image, display

from hdtr import data as D
from hdtr import figures as F
from hdtr.final import compare, file_sha256, load_probs, records_at

def load(name):
    with open(f"{REPO_DIR}/results/{name}", encoding="utf-8") as f:
        return json.load(f)

train, frozen, final = load("06_train.json"), load("07_thresholds.json"), load("08_final_test.json")
print("08 scored at", final["time"], "| final model", final["decisions"]["final_model"],
      "| best baseline", final["decisions"]["best_baseline"])

# %% [markdown]
# ## Figures and tables
# Saved to `MyDrive/hdtr/figures/`: PDF for the LaTeX report, PNG for the slides.

# %%
fig_dir = f"{DRIVE_DIR}/figures"
written = F.build_all(train, frozen, final, fig_dir)
for name, paths in written.items():
    print(name, "→", ", ".join(os.path.basename(p) for p in paths))
for name in written:
    if name.startswith("fig"):
        display(Image(f"{fig_dir}/{name}.png", width=900))

# %%
display(F.main_table(final))
display(F.task_table(final))
display(F.comparisons_table(final))

# %% [markdown]
# ## Exploratory: R2 against every existing detector (final test)
# Same frozen thresholds and prediction files as 08 (each file is checked against its 07 fingerprint),
# same seed-averaged paired source bootstrap. Exploratory: not pre-registered, no multiple-comparison
# correction. Q1 (R2 vs TurkEmbed4STS-HD) stays the only confirmatory comparison.

# %%
systems = frozen["systems"]
changed = [n for n, s in systems.items() if file_sha256(f"{DRIVE_DIR}/predictions/{s['file']}") != s["sha256"]]
if changed:
    raise RuntimeError(f"Predictions changed since thresholds were frozen: {changed}")

tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
tr["source_id"] = signals["source_id"].values
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
test = tr.loc[splits["test"]]
assert len(test) == final["n"]["full"], "test split differs from the one 08 scored"

def runs_of(names):
    return [records_at(test, load_probs(f"{DRIVE_DIR}/predictions/{systems[n]['file']}", list(test.index)), systems[n])
            for n in names]

final_model = final["decisions"]["final_model"]
ours = runs_of(final["systems"]["full"][final_model]["runs"])
existing = [n for n, s in systems.items() if s["kind"] == "existing"]
exploratory, rows = {}, []
for name in existing:
    start = time.perf_counter()
    diff = compare(ours, runs_of([name]), n_boot=N_BOOT)
    exploratory[name] = diff
    for metric, d in diff.items():
        rows.append({"vs": F.NAMES.get(name, name), "metric": metric, "diff": d["diff"], "low": d["low"],
                     "high": d["high"], "CI above 0": d["low"] > 0})
    print(f"{name}: {time.perf_counter() - start:.0f}s")
table = pd.DataFrame(rows)
display(table.pivot(index="vs", columns="metric", values=["diff", "low", "high"]).round(3))

# Same data, thresholds and bootstrap as 08: the comparison with the best baseline must reproduce Q1.
q1 = final["comparisons"]["full"]["Q1"]
for metric in ("example_f1", "span_f1"):
    got = exploratory[final["decisions"]["best_baseline"]][metric]
    assert all(abs(got[k] - q1[metric][k]) < 1e-9 for k in ("diff", "low", "high")), f"{metric} differs from 08's Q1"
print("check passed: the best-baseline comparison reproduces 08's Q1 exactly")

# %% [markdown]
# ## Save
# Send back `MyDrive/hdtr/results/09_exploratory.json`. The figures stay on Drive for the report.

# %%
out = {"notebook": "09_figures", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
       "status": "exploratory, not pre-registered; no multiple-comparison correction",
       "final_model": final_model, "n_boot": N_BOOT, "test_n": len(test),
       "comparisons": {name: {"label": f"{final_model} − {name}", **d} for name, d in exploratory.items()},
       "figures": {k: [os.path.basename(p) for p in v] for k, v in written.items()}}
out_path = f"{DRIVE_DIR}/results/09_exploratory.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("saved", out_path)
