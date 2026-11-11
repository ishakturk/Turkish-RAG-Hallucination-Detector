# colab-runtime: cpu
# %% [markdown]
# # 05a · Native test set: contexts and questions
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# First half of the native Turkish test set (`docs/technical-roadmap.md` §2.7, analysis plan §6):
# 1. **Contexts:** 228 candidates from the 133 Wikipedia articles reserved in `data/mwqa_split.json`
#    (never used for augmentation) and 72 law articles from `mtntasci/turkish-legal-rag` (CC BY 4.0),
#    each at most 1,500 characters. That is 1.5× the final 152 + 48, because some answers will fail the checks in 05b.
# 2. **Design:** every block of four candidates covers the four cells
#    (original or rewritten question × Qwen or DeepSeek answer), fixed by seed 482.
# 3. **Rewritten questions:** for half of the candidates DeepSeek writes a question whose main part
#    the context answers and which also asks for one detail the context lacks. This invites
#    hallucination without making the question unanswerable.
#
# No answers are generated here, and nothing is labeled. Claude reviews the rewritten questions from
# the results file before 05b runs.
#
# **Runtime:** CPU (~5 min, about $0.05). **Secrets:** `GH_TOKEN`, `DEEPSEEK_API_KEY`.
# **Output:** `MyDrive/hdtr/native/candidates.parquet` and `MyDrive/hdtr/results/05a_native_questions.json` (send this one back).

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "cache", "native"):
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
from collections import Counter
import pandas as pd
from openai import AsyncOpenAI

from hdtr import augment as A
from hdtr import native as N

# %% [markdown]
# ## Contexts
# The Wikipedia title split must reproduce the committed one, so no native article was ever used for
# augmentation.

# %%
mwqa = A.load_mwqa_tr(cache=f"{DRIVE_DIR}/cache/mwqa_tr.parquet")
with open(f"{REPO_DIR}/data/mwqa_split.json", encoding="utf-8") as f:
    committed = json.load(f)
assert A.split_titles(mwqa)["native_titles"] == committed["native_titles"], "title split differs from the committed one"

wiki = N.wiki_candidates(mwqa, committed["native_titles"], round(N.SIZES["wikipedia"] * N.OVERSAMPLE))
legal_df = N.load_legal(cache=f"{DRIVE_DIR}/cache/turkish_legal_rag_qa.parquet")
legal = N.legal_candidates(legal_df, round(N.SIZES["legal"] * N.OVERSAMPLE))
cands = N.assign_cells(wiki) + N.assign_cells(legal)

print(f"wikipedia: {len(wiki)} contexts from {len({c['source'] for c in wiki})} articles | legal: {len(legal)} articles")
print("laws:", dict(Counter(c["source"] for c in legal)))
print("cells:", dict(Counter((c["domain"], c["question_type"], c["generator"]) for c in cands)))
lengths = pd.Series([len(N.context_text(c["passages"])) for c in cands])
print("context characters: median", int(lengths.median()), "| max", int(lengths.max()))

# %% [markdown]
# ## Rewritten questions (DeepSeek)
# Resumable: finished calls are kept in a JSONL file on Drive. A rewrite that fails the checks
# (no "?", too long, mentions the passages, same as the original, …) is retried once; a candidate
# whose rewrite still fails is dropped. That is why there are more candidates than needed.

# %%
partial = [c for c in cands if c["question_type"] == "partial"]
jobs = [{"id": c["id"], "messages": N.rewrite_messages(c), "source_question": c["source_question"]} for c in partial]
log_path = f"{DRIVE_DIR}/native/rewrites_v{N.REWRITE_VERSION}.jsonl"

client = AsyncOpenAI(api_key=userdata.get("DEEPSEEK_API_KEY"), base_url=A.BASE_URL)
t0 = time.time()
run = await N.run_api(client, jobs, log_path, N.rewrite_check, json_mode=True, concurrency=10,
                      budget_usd=0.5, max_tokens=300)
print(run, f"| {time.time() - t0:.0f} s")
rewrites = N.latest(A.read_jsonl(log_path))
reasons = Counter(r for line in A.read_jsonl(log_path) for r in line["reasons"])
print("rejection reasons (all attempts):", dict(reasons))

# %% [markdown]
# ## Save candidates

# %%
final_cands = N.with_questions(cands, rewrites)
frame = pd.DataFrame(final_cands)
frame.to_parquet(f"{DRIVE_DIR}/native/candidates.parquet")
print(f"saved {len(frame)} candidates to {DRIVE_DIR}/native/candidates.parquet")
display(frame.groupby(["domain", "question_type", "generator"]).size().rename("candidates").to_frame())
# No questions are printed on purpose: you label these examples later, blind, so it is better not to
# know which questions were rewritten or what detail they probe.

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/05a_native_questions.json` **without opening it**. It lists every rewritten
# question with the detail it probes, so Claude can check that the detail really is missing from the context
# before answers are generated. Reading it would tell you which examples were built to invite hallucination.

# %%
results = {"notebook": "05a_native_questions", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
           "rewrite_version": N.REWRITE_VERSION, "model": A.MODEL, "run": run,
           "rejection_reasons": dict(reasons),
           "candidates": {"wikipedia": len(wiki), "legal": len(legal), "with_question": len(final_cands),
                          "articles": len({c["source"] for c in wiki}),
                          "laws": dict(Counter(c["source"] for c in legal))},
           "cells": {"/".join(k): v for k, v in Counter((c["domain"], c["question_type"], c["generator"])
                                                         for c in final_cands).items()},
           "context_chars": {"median": int(lengths.median()), "max": int(lengths.max())},
           "rewrites": [{"id": c["id"], "domain": c["domain"], "original": c["source_question"],
                         "question": c["question"], "answered_part": c["answered_part"],
                         "missing_detail": c["missing_detail"], "context": N.context_text(c["passages"])}
                        for c in final_cands if c["question_type"] == "partial"],
           "failed_rewrites": [{"id": i, "reasons": l["reasons"], "text": l["text"][:300]}
                               for i, l in rewrites.items() if not l["ok"]]}
out_path = f"{DRIVE_DIR}/results/05a_native_questions.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("saved", out_path)
