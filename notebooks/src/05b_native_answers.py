# %% [markdown]
# # 05b · Native test set: answers and the labeling sheet
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Second half of the native Turkish test set (`docs/technical-roadmap.md` §2.7, analysis plan §6):
# 1. **Answers:** every candidate from 05a gets one answer from its assigned generator,
#    **Qwen3-8B** (on this GPU, thinking off) or **DeepSeek** (API). Both receive the RAGTruth-TR QA
#    prompt the detectors were trained on, as the only user message, the way RAGTruth was built.
# 2. **Checks (objective only, v2):** not cut off at the token limit, not empty, Turkish, not *only* a
#    refusal ("Verilen pasajlara göre cevap verilemiyor."), no thinking text, not too long. A failed answer
#    is regenerated once. Short phrase answers ("9 Eylül 1901") are valid; v1 wrongly rejected them, so
#    answers are re-checked from the logs and each question keeps its first passing attempt.
# 3. **Final set:** per domain and cell, the first candidates (in the fixed 05a order) with an accepted
#    answer, up to 152 Wikipedia + 48 legal = 200. Cells may fall short where a generator refuses too often.
#    Nothing here looks at whether an answer is faithful.
# 4. **Labeling sheet:** `native_labeling.csv`, shuffled, without the generator or the question type.
#
# **Runtime:** L4 GPU on the first run (~20–30 min including the 16 GB Qwen download, about $0.10 of
# DeepSeek). A rerun with every answer already on Drive needs no GPU and finishes in a minute.
# **Secrets:** `GH_TOKEN`, `DEEPSEEK_API_KEY`.
# **Output:** `MyDrive/hdtr/native/native_labeling.csv` (you label this, following `docs/labeling-guide.md`),
# `MyDrive/hdtr/native/native_test.parquet`, and `MyDrive/hdtr/results/05b_native_answers.json` (send this one back).
#
# This notebook prints no answers on purpose: you label them blind, so read them for the first time in the sheet.

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
from collections import Counter
import pandas as pd
import torch, transformers
from openai import AsyncOpenAI

from hdtr import augment as A
from hdtr import native as N
from hdtr.text import words

major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
assert (major, minor) >= (4, 51), f"Qwen3 needs transformers ≥ 4.51, found {transformers.__version__}"

NATIVE = f"{DRIVE_DIR}/native"
cands = pd.read_parquet(f"{NATIVE}/candidates.parquet").to_dict("records")
for c in cands:
    c["passages"] = list(c["passages"])
# Claude's review of the rewritten questions (committed before any answer exists).
with open(f"{REPO_DIR}/{N.REVIEW_FILE}", encoding="utf-8") as f:
    review = json.load(f)
cands, unreviewed = N.apply_review(cands, review)
if unreviewed:
    raise RuntimeError(f"{len(unreviewed)} rewritten questions are not reviewed yet: send "
                       "05a_native_questions.json to Claude first.")
print(len(cands), "candidates after review |", dict(Counter(c["generator"] for c in cands)),
      "| rejected rewrites:", len(review["rejected"]))

# %% [markdown]
# ## DeepSeek answers
# Resumable: finished calls are kept in a JSONL file on Drive and never paid for twice.

# %%
ds_jobs = [{"id": c["id"], "messages": N.answer_messages(c["prompt"])} for c in cands if c["generator"] == "deepseek"]
client = AsyncOpenAI(api_key=userdata.get("DEEPSEEK_API_KEY"), base_url=A.BASE_URL)
t0 = time.time()
ds_run = await N.run_api(client, ds_jobs, f"{NATIVE}/answers_deepseek.jsonl", N.answer_check,
                         concurrency=10, budget_usd=0.5, max_tokens=800)
print(ds_run, f"| {time.time() - t0:.0f} s")

# %% [markdown]
# ## Qwen3-8B answers
# bf16 on the L4, batches of 8, sampling settings from the Qwen3 model card (non-thinking mode).
# Also resumable: after a disconnect, rerun the notebook and only missing answers are generated.

# %%
qw_jobs = [{"id": c["id"], "messages": N.answer_messages(c["prompt"])} for c in cands if c["generator"] == "qwen"]
t0 = time.time()
qw_log = f"{NATIVE}/answers_qwen.jsonl"
# Load the model (16 GB) only if some Qwen answer is still missing; otherwise no GPU is needed.
if N.pending(qw_jobs, A.read_jsonl(qw_log), max_attempts=2):
    assert torch.cuda.is_available(), "Qwen answers are missing and there is no GPU: Runtime → Change runtime type → L4."
    generate = N.qwen_generator()
else:
    generate = None
qw_run = N.run_local(generate, qw_jobs, qw_log, N.answer_check, batch_size=8)
peak = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""
print(qw_run, f"| {time.time() - t0:.0f} s{peak}")

# %% [markdown]
# ## Final set and labeling sheet

# %%
logs = {"deepseek": A.read_jsonl(f"{NATIVE}/answers_deepseek.jsonl"), "qwen": A.read_jsonl(f"{NATIVE}/answers_qwen.jsonl")}
# Answers are re-checked from the logged text with the current checks (v2), taking each question's
# first attempt that passes. No new calls are needed when the checks change.
checked = {g: N.first_passing(lines, N.answer_check) for g, lines in logs.items()}
answers = {**checked["deepseek"], **checked["qwen"]}
final, shortfall = N.select_final(cands, answers)
if shortfall:
    print("WARNING, cells short of their target (send the results file to Claude):", shortfall)

test = pd.DataFrame(final)
test["task_type"] = "QA"
test["labels"] = None          # filled after labeling
test.to_parquet(f"{NATIVE}/native_test.parquet")
sheet = N.labeling_sheet(final)
sheet_path = f"{NATIVE}/native_labeling.csv"
sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
print(f"{len(final)} examples | saved {NATIVE}/native_test.parquet and {sheet_path}")
display(test.groupby(["domain", "question_type", "generator"]).size().rename("examples").to_frame())

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/05b_native_answers.json`. It holds counts only, no answers.
# Then label `native_labeling.csv` following `docs/labeling-guide.md`.

# %%
def answer_stats(rows):
    n = pd.Series([len(words(r["answer"])) for r in rows])
    return {"examples": len(rows), "words_median": int(n.median()), "words_max": int(n.max()),
            "says_context_lacks_info": sum(bool(N._REFUSAL_RE.search(r["answer"])) or "belirtilme" in r["answer"]
                                           for r in rows)}

results = {"notebook": "05b_native_answers", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "models": {"qwen": N.QWEN_MODEL, "qwen_sampling": N.QWEN_SAMPLING, "deepseek": A.MODEL,
                      "deepseek_temperature": N.DEEPSEEK_TEMPERATURE},
           "transformers": transformers.__version__, "runs": {"deepseek": ds_run, "qwen": qw_run},
           "answer_check_version": N.ANSWER_CHECK_VERSION,
           # Per question, under the current checks: accepted, or why its last attempt failed.
           "questions": {g: {"total": len(c), "accepted": sum(l["ok"] for l in c.values()),
                             "accepted_first_try": sum(l["ok"] and l["attempt"] == 1 for l in c.values()),
                             "failed_by_reason": dict(Counter(r for l in c.values() if not l["ok"] for r in l["reasons"])),
                             "accepted_short_answers_le_3_words": sum(l["ok"] and len(words(l["result"])) <= 3
                                                                     for l in c.values())}
                         for g, c in checked.items()},
           "rejection_reasons_v1_log": {g: dict(Counter(r for l in lines for r in l["reasons"])) for g, lines in logs.items()},
           "final": len(final), "shortfall": shortfall,
           "cells": {"/".join(k): v for k, v in Counter((c["domain"], c["question_type"], c["generator"])
                                                         for c in final).items()},
           "answers": {g: answer_stats([c for c in final if c["generator"] == g]) for g in ("qwen", "deepseek")}}
out_path = f"{DRIVE_DIR}/results/05b_native_answers.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("saved", out_path)
