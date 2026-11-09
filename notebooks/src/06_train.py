# %% [markdown]
# # 06 · Training: R0–R4
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Full training runs on all three tasks (QA, Data2txt, Summary). The settings for each run are in
# `configs/r*.yaml`, and the rules for choosing between runs are in `docs/analysis-plan.md`.
#
# | Run | Base model | Recipe | Seeds | Time on L4 (approx.) |
# |---|---|---|---|---|
# | `r0` | TabiBERT | "Newmind-style": 6 fixed epochs, lr 1e-5, plain CE, uncleaned data, no validation | 42 | ~1.5 h |
# | `r1` | TabiBERT (Turkish, 149M) | ours: cleaned data, weighted CE, lr 3e-5, ≤4 epochs, early stopping on validation word F1 | 42, 1234, 2026 | ~1 h each |
# | `r2` | mmBERT-base (multilingual, 307M) | ours | 42, 1234, 2026 | ~1.3 h each |
# | `r3` | LettuceDetect v2 mmBERT (multilingual hallucination training) | ours | 42, 1234, 2026 | ~1.3 h each |
# | `r4` | best of r1–r3 on dev | ours + DeepSeek augmentation (≤25% of examples) | 42, 1234, 2026 | ~1.3–1.7 h each |
#
# **How to run (one group per Colab session is easiest):** set `RUNS` in the next cell, then Run all.
# Suggested order: `["r1"]` → `["r2"]` → `["r3"]` → `["r0"]`. Then send back `06_train.json`; I confirm
# the R4 base, and you run `["r4"]` with `R4_BASE` set. Several groups in one session also works
# (e.g. `["r1", "r2"]`) if you keep the tab open.
#
# **Disconnects:** every finished run is saved on Drive and skipped on the next Run all, so at most the
# run in progress is lost. Keep the tab open (Colab Pro has no background execution).
#
# **What is scored here:** validation (during training) and **dev** (for choosing runs and thresholds).
# Predictions on the final **test** set are saved but **not scored**; that happens once, in notebook 08.
#
# **Output:** per run `MyDrive/hdtr/results/06_runs/<run>.json`, test+dev predictions in
# `MyDrive/hdtr/predictions/`, model weights in `MyDrive/hdtr/models/`, and the summary
# `MyDrive/hdtr/results/06_train.json` — send that back.

# %%
RUNS = ["r1"]                  # any of "r0", "r1", "r2", "r3", "r4"
SEEDS = [42, 1234, 2026]       # r0 always uses seed 42 only
R4_BASE = None                 # "r1", "r2" or "r3": set after R1–R3 are done (Claude confirms which)
KEEP_WEIGHTS = "auto"          # "all", "seed42", or "auto" (all seeds while Drive has > 8 GB free)
LIMIT = None                   # debugging only: cap the number of examples per split

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"
WORK_DIR = "/content/runs"     # checkpoints stay on the local disk; only results and weights go to Drive

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results/06_runs", "predictions", "models", "cache"):
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
import dataclasses, gc, glob, json, pickle, random, shutil, time
import numpy as np
import pandas as pd
import torch

from hdtr import data as D
from hdtr.evaluation import score, tune_thresholds
from hdtr.infer import predict_many
from hdtr.train import RunConfig, train

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(GPU)
if "r4" in RUNS and R4_BASE not in ("r1", "r2", "r3"):
    raise ValueError("Set R4_BASE to 'r1', 'r2' or 'r3' before running r4.")

# %% [markdown]
# ## Data
# RAGTruth-TR with source ids (alignment to English), the fixed splits from `data/splits.json`,
# and, for R4, the augmentation set from notebook 04.

# %%
tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
assert signals["pass"].mean() > 0.99
tr["source_id"] = signals["source_id"].values
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)

def split_frame(name):
    frame = tr.loc[splits[name]]
    return frame.head(LIMIT) if LIMIT else frame

dev = split_frame("dev")
pred_frame = tr.loc[sorted(splits["dev"] + splits["test"])]   # dev + sealed test, predictions only
if LIMIT:
    pred_frame = pred_frame.loc[sorted(set(dev.index) | set(pred_frame.index[:LIMIT]))]
print({k: len(v) for k, v in splits.items() if isinstance(v, list)})

def augmentation_frame(n_train):
    """DeepSeek examples, at most a third of the original train size (≤25% of the mix).
    Faithful/hallucinated pairs stay together; the subset is fixed (seed 482) for all R4 seeds."""
    aug = pd.read_parquet(f"{DRIVE_DIR}/augment/mwqa_aug.parquet")
    aug["labels"] = aug["labels"].map(lambda xs: [dict(x) for x in xs])
    ids = sorted(aug["aug_id"].unique())
    random.Random(482).shuffle(ids)
    cap, keep, n = n_train // 3, [], 0
    sizes = aug.groupby("aug_id").size()
    for i in ids:
        if n + sizes[i] > cap:
            break
        keep.append(i); n += sizes[i]
    return aug[aug["aug_id"].isin(keep)]

# %% [markdown]
# ## Run definitions

# %%
def load_config(group):
    if group == "r4":
        base = RunConfig.from_yaml(f"{REPO_DIR}/configs/{R4_BASE}.yaml")
        return dataclasses.replace(base, name=f"r4_{base.name.split('_', 1)[1]}",
                                   extra={**base.extra, "augment": True, "r4_base": R4_BASE})
    return RunConfig.from_yaml(f"{REPO_DIR}/configs/{group}.yaml")

def planned_runs():
    for group in RUNS:
        base = load_config(group)
        for seed in ([42] if group == "r0" else SEEDS):
            yield group, dataclasses.replace(base, name=f"{base.name}_s{seed}", seed=seed)

for group, cfg in planned_runs():
    print(group, cfg.name, cfg.model_id, f"lr={cfg.lr}", f"epochs={cfg.epochs}", f"loss={cfg.loss}", cfg.extra)

# %% [markdown]
# ## Train, predict, score on dev
# For each run: train (with early stopping where configured), predict dev + test, score dev at
# τ = 0.5 and at the dev-tuned thresholds, save predictions, weights and a result file.

# %%
def drive_free_gb():
    return shutil.disk_usage(DRIVE_DIR).free / 1e9

def keep_weights(cfg):
    if KEEP_WEIGHTS == "all":
        return True
    if KEEP_WEIGHTS == "seed42":
        return cfg.seed == 42
    return cfg.seed == 42 or drive_free_gb() > 8

def frames_for(cfg):
    train_frame = split_frame(cfg.extra["train_split"])
    if cfg.extra.get("augment"):
        aug = augmentation_frame(len(train_frame))
        train_frame = pd.concat([train_frame, aug[["prompt", "answer", "labels", "task_type"]]], ignore_index=True)
    val_split = cfg.extra.get("val_split")
    return train_frame, (split_frame(val_split) if val_split else None)

def eval_history(log_history):
    return [{k: h[k] for k in ("epoch", "step", "eval_loss", "eval_word_f1", "eval_span_f1",
                               "eval_example_f1", "eval_token_f1", "eval_word_auroc") if k in h}
            for h in log_history if "eval_word_f1" in h]

def run_one(group, cfg):
    out_file = f"{DRIVE_DIR}/results/06_runs/{cfg.name}.json"
    if os.path.exists(out_file):
        print(f"{cfg.name}: done already, skipping")
        return
    print(f"\n=== {cfg.name} ({cfg.model_id}) ===")
    train_frame, val_frame = frames_for(cfg)
    attempt, oom_retries = cfg, 0
    while True:
        try:
            run = train(attempt, train_frame, val_frame, f"{WORK_DIR}/{cfg.name}")
            break
        except torch.cuda.OutOfMemoryError:
            if oom_retries == 2 or attempt.batch_size == 1:
                raise
            oom_retries += 1
            attempt = dataclasses.replace(attempt, batch_size=attempt.batch_size // 2,
                                          grad_accum=attempt.grad_accum * 2)
            print(f"out of memory: retrying with batch {attempt.batch_size} x accumulation {attempt.grad_accum}")
            gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(f"{WORK_DIR}/{cfg.name}", ignore_errors=True)

    model, tokenizer, summary = run["model"].eval(), run["tokenizer"], run["summary"]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.autocast("cpu", enabled=False)
    with autocast:
        probs, timing = predict_many(model, tokenizer, zip(pred_frame["prompt"], pred_frame["answer"]),
                                     max_length=cfg.max_length, split_words=cfg.split_words)
    pred_path = f"{DRIVE_DIR}/predictions/06_{cfg.name}.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump({"row_ids": list(pred_frame.index), "probs": probs, "timing": timing,
                     "run": cfg.name, "commit": commit}, f)

    by_id = dict(zip(pred_frame.index, probs))
    dev_probs = [by_id[i] for i in dev.index]
    tuned = tune_thresholds(dev, dev_probs)
    dev_scores = {"tau_0.5": score(dev, dev_probs, 0.5),
                  "tau_example": score(dev, dev_probs, tuned["tau_example"]),
                  "tau_span": score(dev, dev_probs, tuned["tau_span"])}

    weights_dir = None
    if keep_weights(cfg):
        weights_dir = f"{DRIVE_DIR}/models/{cfg.name}"
        model.to(torch.bfloat16).save_pretrained(weights_dir, safe_serialization=True)
        tokenizer.save_pretrained(weights_dir)

    history = eval_history(summary["log_history"])
    record = {
        "name": cfg.name, "group": group, "model_id": cfg.model_id, "seed": cfg.seed, "commit": commit, "gpu": GPU,
        "config": summary["config"], "oom_retries": oom_retries,
        "train_examples": summary["train_examples"], "train_tokens": summary["train_tokens"],
        "truncated_train": summary["truncated_train"], "class_weights": summary["class_weights"],
        "train_minutes": round(summary["train_sec"] / 60, 1), "peak_gpu_gb": summary["peak_gpu_gb"],
        "best_checkpoint": summary["best_checkpoint"],
        "epochs_run": max((h.get("epoch", 0) for h in summary["log_history"]), default=None),
        "validation_history": history,
        "validation_best": max(history, key=lambda h: h["eval_word_f1"]) if history else None,
        "loss_curve": [[h["step"], h["loss"]] for h in summary["log_history"] if "loss" in h],
        "dev": dev_scores, "thresholds": {k: tuned[k] for k in ("tau_example", "tau_span")},
        "threshold_curve": tuned["curve"], "predict_timing": timing,
        "predictions": pred_path, "weights": weights_dir,
    }
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2, default=str)
    d = dev_scores
    print(f"{cfg.name}: {record['train_minutes']} min, {record['epochs_run']} epochs | dev τ0.5 ex-F1 "
          f"{d['tau_0.5']['overall']['example_f1']:.3f} span-F1 {d['tau_0.5']['overall']['span_f1']:.3f} | "
          f"tuned ex-F1 {d['tau_example']['overall']['example_f1']:.3f} (τ {tuned['tau_example']}) "
          f"span-F1 {d['tau_span']['overall']['span_f1']:.3f} (τ {tuned['tau_span']})")

for group, cfg in planned_runs():
    run_one(group, cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(f"{WORK_DIR}/{cfg.name}", ignore_errors=True)

# %% [markdown]
# ## All finished runs on dev
# Mean ± sd over seeds. Selection rule for R4 (`docs/analysis-plan.md` §4): among R1–R3 with all seeds
# done, the highest mean of (example F1 at τ_example + span F1 at τ_span) / 2; a gap under 0.01 goes to
# the smaller model.

# %%
records = []
for path in sorted(glob.glob(f"{DRIVE_DIR}/results/06_runs/*.json")):
    with open(path) as f:
        records.append(json.load(f))

rows = []
for r in records:
    d = r["dev"]
    rows.append({"group": r["group"], "run": r["name"], "seed": r["seed"],
                 "ex_f1@0.5": d["tau_0.5"]["overall"]["example_f1"], "span_f1@0.5": d["tau_0.5"]["overall"]["span_f1"],
                 "ex_f1@τ": d["tau_example"]["overall"]["example_f1"], "span_f1@τ": d["tau_span"]["overall"]["span_f1"],
                 "ex_auroc": d["tau_0.5"]["overall"]["example_auroc"], "word_auroc": d["tau_0.5"]["overall"]["word_auroc"],
                 **{f"span_f1@τ {t}": d["tau_span"].get(t, {}).get("span_f1") for t in ("QA", "Data2txt", "Summary")},
                 "val_word_f1": (r["validation_best"] or {}).get("eval_word_f1"),
                 "epochs": r["epochs_run"], "minutes": r["train_minutes"]})
table = pd.DataFrame(rows)
if len(table):
    display(table.set_index("run").round(3))
    metrics_cols = [c for c in table.columns if c not in ("group", "run", "seed")]
    summary_table = table.groupby("group")[metrics_cols].agg(["mean", "std"]).round(3)
    display(summary_table)

selection = None
complete = {g: t for g, t in table.groupby("group") if g in ("r1", "r2", "r3") and len(t) == len(SEEDS)} if len(table) else {}
if complete:
    params = {"r1": 149, "r2": 307, "r3": 307}
    scores = {g: float(((t["ex_f1@τ"] + t["span_f1@τ"]) / 2).mean()) for g, t in complete.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    for g in ranked[1:]:
        if scores[best] - scores[g] < 0.01 and params[g] < params[best]:
            best = g
    selection = {"scores": scores, "r4_base": best, "complete_groups": sorted(complete),
                 "final_once_all_done": len(complete) == 3}
    print("Selection score per group:", {g: round(s, 4) for g, s in scores.items()})
    print("R4 base by the rule:", best, "" if len(complete) == 3 else "(provisional: not all of R1–R3 are done)")

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/06_train.json`.

# %%
slim = [{k: v for k, v in r.items() if k not in ("threshold_curve", "loss_curve")} for r in records]
results = {"notebook": "06_train", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"), "gpu": GPU,
           "runs_requested": RUNS, "seeds": SEEDS, "r4_base": R4_BASE, "selection": selection,
           "runs": slim, "loss_curves": {r["name"]: r["loss_curve"] for r in records},
           "threshold_curves": {r["name"]: r["threshold_curve"] for r in records}}
out_path = f"{DRIVE_DIR}/results/06_train.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print("saved", out_path, f"({len(records)} runs) | Drive free: {drive_free_gb():.1f} GB")
