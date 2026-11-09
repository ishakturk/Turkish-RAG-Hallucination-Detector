# %% [markdown]
# # 03 · Pilot: which base model and which recipe?
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Short, cheap runs to decide what gets a full training run. Every run trains for **1 epoch on the
# QA part** of the cleaned training set and is scored on the **QA part of the validation set**
# (carved from train; the dev/test sets are not touched here).
#
# | Run | Base model | What it tests |
# |---|---|---|
# | `tabibert_base` | TabiBERT (Turkish, 149M) | our default recipe: lr 2e-5, class-weighted CE |
# | `tabibert_lr1e-5`, `tabibert_lr3e-5` | TabiBERT | learning rate |
# | `tabibert_focal` | TabiBERT | focal loss instead of weighted CE |
# | `tabibert_splitwords` | TabiBERT | word-aligned tokens (9% of TabiBERT tokens span two words, e.g. "a kadar") |
# | `moganbert_tr` | MoganBERT-TR (Turkish, 149M, Aug 2026) | alternative Turkish encoder |
# | `mmbert_small` | mmBERT-small (multilingual, 140M) | small multilingual base |
# | `mmbert_base` | mmBERT-base (multilingual, 307M) | multilingual base (R2 candidate) |
# | `krlabs_v2_mmbert` | LettuceDetect v2 mmBERT | multilingual *hallucination* pretraining (R3 candidate) |
#
# **Runtime:** L4 GPU, ~60–80 min in total. Finished runs are skipped if you rerun after a disconnect.
# **Output:** `MyDrive/hdtr/results/03_pilot.json` — send it back.

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results/03_pilot_runs", "cache"):
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
import gc, json, shutil, time
import numpy as np
import pandas as pd
import torch

from hdtr import data as D
from hdtr.infer import predict_many
from hdtr.metrics import evaluate
from hdtr.postprocess import char_probs_to_spans
from hdtr.train import RunConfig, train

print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## Data: QA part of the cleaned train and validation splits

# %%
tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
assert signals["pass"].mean() > 0.99
tr["source_id"] = signals["source_id"].values
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)

train_qa = tr.loc[splits["train_clean"]].query("task_type == 'QA'")
val_qa = tr.loc[splits["validation_clean"]].query("task_type == 'QA'")
print(f"train QA: {len(train_qa)} | validation QA: {len(val_qa)} "
      f"(hallucinated share {val_qa['labels'].map(len).gt(0).mean():.2f})")

# %% [markdown]
# ## Pilot runs
# Batch sizes keep the effective batch at 16 for every model.

# %%
TABI = "boun-tabilab/TabiBERT"
RUNS = [
    RunConfig("tabibert_base", TABI),
    RunConfig("tabibert_lr1e-5", TABI, lr=1e-5),
    RunConfig("tabibert_lr3e-5", TABI, lr=3e-5),
    RunConfig("tabibert_focal", TABI, loss="focal"),
    RunConfig("tabibert_splitwords", TABI, split_words=True),
    RunConfig("moganbert_tr", "moganai/MoganBERT-TR"),
    RunConfig("mmbert_small", "jhu-clsp/mmBERT-small"),
    RunConfig("mmbert_base", "jhu-clsp/mmBERT-base", batch_size=4, grad_accum=4),
    RunConfig("krlabs_v2_mmbert", "KRLabsOrg/lettucedect-v2-mmbert-base", batch_size=4, grad_accum=4),
]

def validate(model, tokenizer, frame, split_words):
    """Our metrics on validation: τ = 0.5, plus the best example-F1 threshold (diagnostic only)."""
    probs, timing = predict_many(model, tokenizer, zip(frame["prompt"], frame["answer"]),
                                 max_length=4096, split_words=split_words)
    def records(th):
        return [{"answer": a, "gold_spans": g, "pred_spans": char_probs_to_spans(a, p, th),
                 "char_probs": p, "task_type": t, "group": s}
                for a, g, p, t, s in zip(frame["answer"], frame["labels"], probs,
                                         frame["task_type"], frame["source_id"])]
    m = evaluate(records(0.5), by_task=False)["overall"]
    sweep = {th: evaluate(records(th), by_task=False)["overall"] for th in (0.3, 0.4, 0.6, 0.7)}
    best_th = max([0.5, *sweep], key=lambda th: (m if th == 0.5 else sweep[th])["span"]["f1"])
    best = m if best_th == 0.5 else sweep[best_th]
    return {
        "example_f1": m["example"]["f1"], "word_f1": m["word"]["f1"], "span_f1": m["span"]["f1"],
        "example_auroc": m.get("example_auroc"), "word_auroc": m.get("word_auroc"),
        "best_threshold": best_th, "best_span_f1": best["span"]["f1"], "best_example_f1": best["example"]["f1"],
        "sec_per_example": timing["sec_per_example"],
    }

for cfg in RUNS:
    out_file = f"{DRIVE_DIR}/results/03_pilot_runs/{cfg.name}.json"
    if os.path.exists(out_file):
        print(f"{cfg.name}: done already, skipping")
        continue
    print(f"\n=== {cfg.name} ===")
    work_dir = f"/content/runs/{cfg.name}"
    try:
        run = train(cfg, train_qa, None, work_dir)
        run["model"].eval()
        val = validate(run["model"], run["tokenizer"], val_qa, cfg.split_words)
        summary = run["summary"]
        losses = [h["loss"] for h in summary["log_history"] if "loss" in h]
        record = {
            "name": cfg.name, "model_id": cfg.model_id, "config": summary["config"],
            "train_examples": summary["train_examples"], "train_tokens": summary["train_tokens"],
            "train_minutes": round(summary["train_sec"] / 60, 2),
            "tokens_per_sec": round(summary["train_tokens"] / summary["train_sec"]),
            "peak_gpu_gb": summary["peak_gpu_gb"], "class_weights": summary["class_weights"],
            "loss_first_last": [losses[0], losses[-1]] if losses else None,
            "loss_curve": losses, "validation_qa": val,
        }
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in val.items()},
              f"| {record['train_minutes']} min, {record['peak_gpu_gb']} GB")
    except Exception as exc:
        record = {"name": cfg.name, "model_id": cfg.model_id, "error": f"{type(exc).__name__}: {exc}"[:800]}
        print("FAILED:", record["error"])
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2, default=str)
    run = None  # free the model before the next run
    shutil.rmtree(work_dir, ignore_errors=True)
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Comparison
# Validation QA, threshold 0.5 (and the best of a small threshold sweep, diagnostic only).
# Extrapolated full-run time = minutes per QA epoch × (all-task tokens / QA tokens) × epochs.

# %%
records = []
for cfg in RUNS:
    with open(f"{DRIVE_DIR}/results/03_pilot_runs/{cfg.name}.json") as f:
        records.append(json.load(f))
table = pd.DataFrame([
    {"run": r["name"], **({k: r["validation_qa"][k] for k in
       ("example_f1", "span_f1", "example_auroc", "word_auroc", "best_threshold", "best_span_f1")}
      if "validation_qa" in r else {"error": r.get("error", "")[:60]}),
     "train_min": r.get("train_minutes"), "peak_GB": r.get("peak_gpu_gb")}
    for r in records
]).set_index("run")
display(table.round(4))

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/03_pilot.json`.

# %%
all_train = tr.loc[splits["train_clean"]]
results = {"notebook": "03_pilot", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "gpu": torch.cuda.get_device_name(0), "train_qa": len(train_qa), "val_qa": len(val_qa),
           "train_clean_total": len(all_train), "runs": records}
out_path = f"{DRIVE_DIR}/results/03_pilot.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print("saved", out_path)
