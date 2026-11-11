# %% [markdown]
# # 11 · Release: native test set, native predictions, final model
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Prepares everything the deliverable notebook (`main.ipynb`) needs, so that it runs for anyone
# without Google Drive or a GPU-heavy training run. Nothing is retrained or rescored here.
#
# 1. **Native test set** → `MyDrive/hdtr/release/native_test.jsonl` (184 answers with label and gold spans).
# 2. **Native predictions** of all 22 runs → `MyDrive/hdtr/release/native_predictions.npz`, checked by
#    recomputing the native results and comparing them with `results/10_native_test.json`.
# 3. **Final model** (R2 mmBERT-base, the seed with the best dev selection score: `r2_mmbert_s42`) with a
#    model card, uploaded to a **private** Hugging Face repository. It is made public on submission day.
#    The upload is checked by downloading the model again and predicting the native set.
#
# **Runtime:** L4 GPU, ~5–10 min. **Secrets:** `GH_TOKEN` (read the repo) and `HF_TOKEN`
# (Hugging Face → Settings → Access Tokens → a token with **Write** permission). Without `HF_TOKEN`
# steps 1–2 still run and step 3 is skipped.
# **Send back:** `native_test.jsonl`, `native_predictions.npz` and `release.json` from `MyDrive/hdtr/release/`.

# %%
RELEASE_RUN = "r2_mmbert_s42"
HF_REPO_NAME = "turkish-rag-hallucination-detector"   # created under your Hugging Face account

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"
OUT_DIR = f"{DRIVE_DIR}/release"

from google.colab import drive, userdata
drive.mount("/content/drive")
os.makedirs(OUT_DIR, exist_ok=True)

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
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip().replace(token, '***')}")
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
import json, pickle, shutil, time
import numpy as np
import pandas as pd
import torch

from hdtr import native_eval as NE
from hdtr import reproduce as R
from hdtr.final import file_sha256
from hdtr.infer import load_detector, predict_many

with open(f"{REPO_DIR}/results/07_thresholds.json", encoding="utf-8") as f:
    frozen = json.load(f)
with open(f"{REPO_DIR}/results/10_native_test.json", encoding="utf-8") as f:
    native_results = json.load(f)
with open(f"{REPO_DIR}/results/08_final_test.json", encoding="utf-8") as f:
    final_results = json.load(f)
systems = frozen["systems"]
assert RELEASE_RUN == R.RELEASE_RUN
best = max((n for n, s in systems.items() if s.get("group") == "r2"), key=lambda n: systems[n]["selection_score"])
assert best == RELEASE_RUN, f"the best R2 seed on dev is {best}, not {RELEASE_RUN}"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# %% [markdown]
# ## 1 · Native test set
# Built exactly as in notebook 10 (`native_frame` checks ids, answers and span offsets), then written
# one answer per line.

# %%
reference = pd.read_csv(f"{REPO_DIR}/data/native_reference_labels.csv")
with open(f"{REPO_DIR}/data/native_spans.json", encoding="utf-8") as f:
    spans = json.load(f)["spans"]
sheet = pd.read_csv(f"{REPO_DIR}/data/native_labeling_blank.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
native = NE.native_frame(pd.read_parquet(f"{DRIVE_DIR}/native/native_test.parquet"), reference, spans,
                         dict(zip(sheet["id"], sheet["answer"])))
R.write_jsonl(R.native_records(native), f"{OUT_DIR}/native_test.jsonl")
again = R.load_native_test(f"{OUT_DIR}/native_test.jsonl")
for col in ("answer", "prompt", "labels", "reference_label", "task_type", "source_id"):
    assert list(again[col]) == list(native[col]), col
print(len(again), "answers |", int(again["reference_label"].sum()), "hallucinated | written native_test.jsonl")
display(again.groupby(["domain", "generator", "question_type"])["label"].agg(["sum", "count"]))

# %% [markdown]
# ## 2 · Native predictions of every run
# The float16 predictions notebook 10 saved, packed into one file. Recomputing the native results from
# this file must give exactly the numbers in `results/10_native_test.json`.

# %%
probs_by_system = {}
for name in systems:
    with open(f"{DRIVE_DIR}/predictions/10_native_{name}.pkl", "rb") as f:
        saved = pickle.load(f)
    assert saved["row_ids"] == list(native.index), name
    probs_by_system[name] = saved["probs"]
np.savez_compressed(f"{OUT_DIR}/native_predictions.npz", **R.pack_predictions(list(native.index), probs_by_system))
loaded = R.load_native_predictions(f"{OUT_DIR}/native_predictions.npz", again)
check = R.compare_with_saved(R.rescore(again, loaded, systems), native_results)
print(f"{len(loaded)} runs packed, {os.path.getsize(f'{OUT_DIR}/native_predictions.npz') / 1e6:.1f} MB |",
      "largest difference from 10_native_test.json:", check["abs_diff"].max())
assert check["abs_diff"].max() < 1e-9

# %% [markdown]
# ## 3 · Final model on Hugging Face
# Weights from `MyDrive/hdtr/models/r2_mmbert_s42` (bf16, as scored), a model card with the frozen
# thresholds and the scores of this exact run, then a private upload.

# %%
s = systems[RELEASE_RUN]
run_final = final_results["per_run"][RELEASE_RUN]["full"]["tuned"]["overall"]
run_native = native_results["per_run"][RELEASE_RUN]["tuned"]["overall"]
MODEL_DIR = f"/content/release_model"
shutil.copytree(f"{DRIVE_DIR}/models/{RELEASE_RUN}", MODEL_DIR, dirs_exist_ok=True)

card = f"""---
language: [tr]
license: mit
base_model: jhu-clsp/mmBERT-base
pipeline_tag: token-classification
tags: [hallucination-detection, rag, turkish, token-classification]
---

# Turkish RAG hallucination detector (mmBERT-base)

Marks the parts of a Turkish RAG answer that are **not supported by the given context**.
BIL482 term project (İstanbul Medeniyet University, 2026). Code, data splits, evaluation and report:
https://github.com/{REPO}

- **Base model:** `jhu-clsp/mmBERT-base` (307M), fine-tuned for token classification on the cleaned
  RAGTruth-TR training split (run `{RELEASE_RUN}`, weighted cross-entropy, lr 3e-5, early stopping).
- **Input:** `[CLS] prompt [SEP] answer [SEP]`; the prompt holds the question or instruction and the context.
  Label 1 = hallucinated token of the answer.
- **Thresholds (frozen on the dev split):** answer-level τ = {s['tau_example']}, span-level τ = {s['tau_span']}.
  A word is flagged when its highest character probability reaches τ.

## Scores of this run

| Test set | Example F1 | Example AUROC | Span F1 |
|---|---|---|---|
| RAGTruth-TR final test (2,160 translated answers) | {run_final['example_f1']:.3f} | {run_final['example_auroc']:.3f} | {run_final['span_f1']:.3f} |
| Native Turkish test (184 answers by Qwen3-8B and DeepSeek) | {run_native['example_f1']:.3f} | {run_native['example_auroc']:.3f} | {run_native['span_f1']:.3f} |

Performance drops on natural Turkish answers: the model is precise but misses many short, subtle errors.
See the report in the repository for the full comparison with seven existing Turkish detectors.

## Use

```python
# pip install "git+https://github.com/{REPO}"
from hdtr.infer import load_detector
from hdtr.reproduce import detect
tok, model = load_detector("<this repo id>")
out = detect(model, tok, question, passages, answer, threshold={s['tau_span']})
print(out["spans"])
```

## Limitations

Trained on machine-translated data; native Turkish errors such as a wrong year inferred from a nearby
number are often missed. Labels of the native test set are LLM-made and validated on a 50-row human sample.
"""
with open(f"{MODEL_DIR}/README.md", "w", encoding="utf-8") as f:
    f.write(card)
print(card[:600], "...")

# %%
hf_token = userdata.get("HF_TOKEN")
release = {"notebook": "11_release", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "release_run": RELEASE_RUN, "thresholds": {"tau_example": s["tau_example"], "tau_span": s["tau_span"]},
           "files": {n: file_sha256(f"{OUT_DIR}/{n}") for n in ("native_test.jsonl", "native_predictions.npz")},
           "hf_repo": None, "upload_check": None}
if not hf_token:
    print("HF_TOKEN is missing: the model was not uploaded. Add the secret and run this cell again.")
else:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    repo_id = f"{api.whoami()['name']}/{HF_REPO_NAME}"
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=MODEL_DIR, repo_id=repo_id, commit_message=f"{RELEASE_RUN} from commit {commit}")
    release["hf_repo"] = repo_id
    print("uploaded to https://huggingface.co/" + repo_id, "(private)")

    # Download it again and predict the native set the way notebook 10 did.
    tok, model = load_detector(repo_id, token=hf_token)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        probs, _ = predict_many(model, tok, zip(again["prompt"], again["answer"]), max_length=4096)
    diff = max(float(np.abs(p - q).max()) for p, q in zip(probs, loaded[RELEASE_RUN]) if len(p))
    release["upload_check"] = {"max_abs_prob_diff_vs_notebook_10": diff}
    print(f"downloaded model vs notebook 10 predictions: max |Δp| = {diff:.4f} (should be small, < 0.05)")

with open(f"{OUT_DIR}/release.json", "w", encoding="utf-8") as f:
    json.dump(release, f, ensure_ascii=False, indent=2)
print(json.dumps(release, indent=2))
print("Send back native_test.jsonl, native_predictions.npz and release.json from", OUT_DIR)
