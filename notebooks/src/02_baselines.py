# %% [markdown]
# # 02 · Baselines: existing detectors under one protocol
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Every existing Turkish (or multilingual) hallucination detector is run on the same 2,700
# RAGTruth-TR test responses with the same code, and scored with the same metrics
# (`hdtr.metrics`: example, word and span level, AUROC from probabilities).
#
# **Test discipline.** Predictions are saved for all 2,700 test responses, but in this notebook
# we only *score* the **dev** part (540 responses, 90 sources). The final 2,160-response test part
# stays sealed until the final evaluation (notebook 08), so no decision is tuned on it.
#
# Systems:
# - 6 Newmind detectors (Turk-LettuceDetect paper + newer releases)
# - KR Labs `lettucedect-v2-mmbert-base` (multilingual, never trained on Turkish → zero-shot)
# - 2 classical baselines (lexical overlap, word-level logistic regression)
#
# **Runtime:** L4 GPU, ~25–35 min. **Secrets:** `GH_TOKEN`.
# **Output:** `MyDrive/hdtr/results/02_baselines_dev.json` — send it back.
# Predictions are stored in `MyDrive/hdtr/predictions/` for later notebooks.

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "predictions", "cache"):
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
from hdtr.metrics import evaluate
from hdtr.postprocess import char_probs_to_spans

results = {"notebook": "02_baselines", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

# %% [markdown]
# ## Data
# The grouped splits were built in notebook 01 and are committed in `data/splits.json`.
# `source_id` comes from aligning each Turkish row to the English RAGTruth (99.99% of rows align).

# %%
tr = D.load_ragtruth_tr()
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
signals = D.align_by_order(tr, en)
assert signals["pass"].mean() > 0.99, "alignment changed: rerun notebook 01"
tr["source_id"] = signals["source_id"].values

with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
dev = tr.loc[splits["dev"]]
test_all = tr.loc[sorted(splits["dev"] + splits["test"])]  # predictions only; test part is not scored here
print("dev:", len(dev), "| predicted (dev + sealed test):", len(test_all))
print(dev.groupby("task_type").size())

def score(frame, probs, threshold=0.5):
    """Metrics for one system on one frame, from per-character probabilities."""
    records = [
        {"answer": a, "gold_spans": g, "pred_spans": char_probs_to_spans(a, p, threshold),
         "char_probs": p, "task_type": t, "group": s}
        for a, g, p, t, s in zip(frame["answer"], frame["labels"], probs, frame["task_type"], frame["source_id"])
    ]
    return evaluate(records)

def compact(ev):
    """Keep the headline numbers per task for the JSON report."""
    return {task: {
        "example_f1": round(m["example"]["f1"], 4), "example_p": round(m["example"]["precision"], 4),
        "example_r": round(m["example"]["recall"], 4), "example_macro_f1": round(m["example"]["macro_f1"], 4),
        "word_f1": round(m["word"]["f1"], 4), "word_macro_f1": round(m["word"]["macro_f1"], 4),
        "span_f1": round(m["span"]["f1"], 4), "span_p": round(m["span"]["precision"], 4),
        "span_r": round(m["span"]["recall"], 4),
        "example_auroc": round(m.get("example_auroc", float("nan")), 4),
        "word_auroc": round(m.get("word_auroc", float("nan")), 4), "n": m["n"],
    } for task, m in ev.items()}

# %% [markdown]
# ## Transformer detectors
# Each model reads `[CLS] prompt [SEP] answer [SEP]` (prompt = instruction + context + question),
# outputs a hallucination probability per answer token, and we spread it over the characters the
# token covers. Spans are then formed on Turkish word boundaries (`hdtr.postprocess`).
# Threshold 0.5 here; per-system thresholds are tuned on dev in notebook 07.

# %%
from hdtr.infer import load_detector, predict_many

DETECTORS = {
    "newmind_modernbert_tr": ("newmindai/ModernBERT-tr-uncased-stsb-HD", False),
    "newmind_turkembed4sts": ("newmindai/TurkEmbed4STS-HD", True),
    "newmind_eurobert_tr": ("newmindai/lettucedect-210m-eurobert-tr-v1", True),
    "newmind_bge_m3": ("newmindai/bge-m3-HD", False),
    "newmind_ettin_150m": ("newmindai/ettin-encoder-150M-TR-HD", False),
    "newmind_ettin_32m": ("newmindai/ettin-encoder-32M-TR-HD", False),
    "krlabs_v2_mmbert_zeroshot": ("KRLabsOrg/lettucedect-v2-mmbert-base", False),
}
results["systems"] = {}
pairs = list(zip(test_all["prompt"], test_all["answer"]))
dev_pos = [test_all.index.get_loc(i) for i in dev.index]

for name, (model_id, remote_code) in DETECTORS.items():
    pred_path = f"{DRIVE_DIR}/predictions/{name}.pkl"
    try:
        if os.path.exists(pred_path):  # resume after a disconnect
            with open(pred_path, "rb") as f:
                saved = pickle.load(f)
            probs, timing = saved["probs"], saved["timing"]
        else:
            kwargs = {"trust_remote_code": True} if remote_code else {}
            tok, model = load_detector(model_id, **kwargs)
            probs, timing = predict_many(model, tok, pairs, max_length=8192)
            probs = [p.astype(np.float16) for p in probs]
            with open(pred_path, "wb") as f:
                pickle.dump({"row_ids": list(test_all.index), "probs": probs, "timing": timing,
                             "model_id": model_id, "id2label": model.config.id2label}, f)
            del model
            gc.collect(); torch.cuda.empty_cache()
        dev_probs = [probs[i].astype(np.float32) for i in dev_pos]
        results["systems"][name] = {"model_id": model_id, "timing": timing,
                                    "dev_t0.5": compact(score(dev, dev_probs))}
        o = results["systems"][name]["dev_t0.5"]["overall"]
        print(f"{name:28s} ex-F1 {o['example_f1']:.3f}  span-F1 {o['span_f1']:.3f}  "
              f"ex-AUROC {o['example_auroc']:.3f}  {timing['sec_per_example']:.3f}s/ex")
    except Exception as exc:  # one broken model must not stop the others
        results["systems"][name] = {"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"[:500]}
        print(f"{name}: FAILED — {results['systems'][name]['error']}")
        gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Classical baselines (CPU)
# - **Lexical overlap:** an answer sentence that matches no context sentence (character n-gram TF-IDF) is flagged.
# - **Word logistic regression:** features such as "is this word / its first 5 letters / this number in the context",
#   trained on 4,000 cleaned training examples.

# %%
from hdtr.baselines import LexicalOverlap, WordLogReg

train = tr.loc[splits["train_clean"]]
fit_sample = train.sample(4000, random_state=0)
lex = LexicalOverlap().fit(train["prompt"].sample(3000, random_state=0), train["answer"].sample(3000, random_state=0))
logreg = WordLogReg(lex).fit(fit_sample["prompt"], fit_sample["answer"], fit_sample["labels"])
results["word_logreg_coefficients"] = logreg.coefficients()

for name, model in {"classical_lexical_overlap": lex, "classical_word_logreg": logreg}.items():
    start = time.perf_counter()
    probs = [model.predict_char_probs(p, a).astype(np.float16) for p, a in pairs]
    timing = {"n": len(probs), "sec_per_example": (time.perf_counter() - start) / len(probs)}
    with open(f"{DRIVE_DIR}/predictions/{name}.pkl", "wb") as f:
        pickle.dump({"row_ids": list(test_all.index), "probs": probs, "timing": timing}, f)
    dev_probs = [probs[i].astype(np.float32) for i in dev_pos]
    results["systems"][name] = {"timing": timing, "dev_t0.5": compact(score(dev, dev_probs))}
    o = results["systems"][name]["dev_t0.5"]["overall"]
    print(f"{name:28s} ex-F1 {o['example_f1']:.3f}  span-F1 {o['span_f1']:.3f}  ex-AUROC {o['example_auroc']:.3f}")

# %% [markdown]
# ## Dev summary
# Example-level and span-level F1 (hallucinated class) and example AUROC per system and task.
# Remember: dev only, fixed threshold 0.5. These are not the final numbers.

# %%
rows = []
for name, r in results["systems"].items():
    if "dev_t0.5" not in r:
        continue
    for task, m in r["dev_t0.5"].items():
        rows.append({"system": name, "task": task, "ex_F1": m["example_f1"], "span_F1": m["span_f1"],
                     "word_F1": m["word_f1"], "ex_AUROC": m["example_auroc"]})
summary = pd.DataFrame(rows).pivot(index="system", columns="task", values=["ex_F1", "span_F1", "ex_AUROC"])
display(summary.round(3))

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/02_baselines_dev.json`.

# %%
out_path = f"{DRIVE_DIR}/results/02_baselines_dev.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("saved", out_path)
