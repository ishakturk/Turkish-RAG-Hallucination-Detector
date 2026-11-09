# %% [markdown]
# # 01 · Data audit and splits
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# RAGTruth-TR is a machine translation of RAGTruth that was never validated. Before training on it we:
#
# 1. Describe it: splits, tasks, label ratio, label types, lengths
# 2. Align every Turkish row to its English original (this gives us the `source_id` that RAGTruth-TR lacks)
# 3. Audit translation quality with fixed rules and count what would be removed
# 4. Measure input lengths under each tokenizer (how much would 512-token models truncate?)
# 5. Build grouped splits: train / validation / dev-from-test / final test
#
# **Runtime:** CPU is enough (GPU not needed). **Secrets:** `GH_TOKEN`.
# **Output:** `MyDrive/hdtr/results/01_audit.json` and `MyDrive/hdtr/data/splits.json` — send both back.

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "data", "cache"):
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
import numpy as np
import pandas as pd

from hdtr import data as D

results = {"notebook": "01_data_audit", "commit": commit, "time": time.strftime("%Y-%m-%d %H:%M")}

def short(text, n=160):
    """Shorten long texts for the JSON report."""
    return text if len(text) <= n else text[:n] + "…"

# %% [markdown]
# ## 1. Describe RAGTruth-TR
# The Hugging Face dataset has a single split; the real train/test split is the `split` column.

# %%
tr = D.load_ragtruth_tr()
tr["n_spans"] = tr["labels"].map(len)
tr["hallucinated"] = tr["n_spans"] > 0

crosstab = pd.crosstab(tr["split"], tr["task_type"], margins=True)
halluc_share = tr.groupby(["split", "task_type"])["hallucinated"].mean().round(3)
label_types = pd.Series([l["label"] for ls in tr["labels"] for l in ls]).value_counts()
lengths = tr.assign(prompt_chars=tr["prompt"].str.len(), answer_chars=tr["answer"].str.len())[
    ["prompt_chars", "answer_chars"]].describe(percentiles=[.5, .9, .99]).round(0)

display(crosstab, halluc_share.unstack(), label_types, lengths)
results["describe"] = {
    "crosstab": crosstab.to_dict(),
    "hallucinated_share": {f"{s}/{t}": v for (s, t), v in halluc_share.items()},
    "label_types": label_types.to_dict(),
    "lengths_chars": lengths.to_dict(),
    "spans_per_answer_mean": float(tr["n_spans"].mean()),
}

# %% [markdown]
# ## 2. Align to the English RAGTruth
# Hypothesis: the Turkish rows were translated in the same order as the English response file
# (as preprocessed by LettuceDetect). For each candidate pairing we check, row by row:
# same split, same task, length ratio 0.6–1.8, and both labeled / both clean. Digit agreement and
# exact span counts are reported too (translation legitimately rewrites some numbers, e.g.
# "5 pm" → "17:00"). We accept a pairing if ≥ 95% of rows pass.

# %%
en = D.load_ragtruth_en(cache_dir=f"{DRIVE_DIR}/cache")
print("English rows:", len(en))
display(pd.crosstab(en["split"], en["task_type"], margins=True))

attempts = {}
for name, by in {"global_order": (), "order_within_split_task": ("split", "task_type")}.items():
    try:
        signals = D.align_by_order(tr, en, by=by)
        attempts[name] = {
            "pass_rate": float(signals["pass"].mean()),
            "same_n_spans_rate": float(signals["same_n_spans"].mean()),
            "digits_agree_rate": float(signals["digits_agree"].mean()),
            "len_ratio_median": float(signals["len_ratio"].median()),
            "_signals": signals,
        }
    except ValueError as exc:
        attempts[name] = {"error": str(exc)}
    print(name, {k: v for k, v in attempts[name].items() if not k.startswith("_")})

best_name = max((n for n in attempts if "pass_rate" in attempts[n]),
                key=lambda n: attempts[n]["pass_rate"], default=None)
aligned = best_name is not None and attempts[best_name]["pass_rate"] >= 0.95
print("\nbest pairing:", best_name, "| accepted:", aligned)

results["alignment"] = {n: {k: v for k, v in a.items() if not k.startswith("_")} for n, a in attempts.items()}
results["alignment"]["best"] = best_name
results["alignment"]["accepted"] = aligned

# %% [markdown]
# Rows that failed the checks, for manual inspection (the first 8).

# %%
if best_name:
    signals = attempts[best_name]["_signals"]
    failed = signals[~signals["pass"]].head(8)
    en_by_id = en.set_index("response_id")
    examples = []
    for _, s in failed.iterrows():
        tr_row = tr.loc[s["row_id"]]
        en_row = en_by_id.loc[s["response_id"]]
        examples.append({
            "row_id": int(s["row_id"]), "task": tr_row["task_type"],
            "n_spans_tr": int(s["n_spans_tr"]), "n_spans_en": int(s["n_spans_en"]),
            "len_ratio": round(float(s["len_ratio"]), 2), "digits_agree": bool(s["digits_agree"]),
            "tr": short(tr_row["answer"]), "en": short(en_row["answer"]),
        })
    results["alignment"]["failed_examples"] = examples
    for e in examples[:4]:
        print(json.dumps(e, ensure_ascii=False, indent=1))

# %% [markdown]
# ## 3. Translation audit
# Rules (fixed before training; see `docs/technical-roadmap.md` §2.5):
#
# | Flag | Type |
# |---|---|
# | span outside the answer / empty span | hard drop |
# | span count differs from English | hard drop |
# | degenerate repetition: a 4-gram ≥ 6 times and > 2× the English original's max | hard drop |
# | answer truncated while the English one is complete | hard drop |
# | answer length ratio TR/EN outside 0.5–2.0 | soft |
# | > 15% English function words | soft |
# | translation added a number found in neither the English answer nor the context (e.g. a °C conversion) | soft |
# | span starts or ends inside a word | soft (fixed later by word snapping) |
#
# Soft flags only drop a row when combined with another flag.

# %%
if aligned:
    signals = attempts[best_name]["_signals"].set_index("row_id")
    en_by_id = en.set_index("response_id")
    flag_rows = []
    for row in tr.itertuples():
        en_row = en_by_id.loc[signals.loc[row.row_id, "response_id"]]
        flags = D.audit_row(row.answer, row.labels, en_row["answer"], en_row["labels"], row.prompt)
        flag_rows.append({"row_id": row.row_id, **flags, "drop": D.drop_decision(flags)})
    tr = tr.join(signals[["source_id"]], on="row_id")
else:
    # Without alignment only the Turkish-side rules can run.
    flag_rows = [{"row_id": r.row_id, **D.audit_row(r.answer, r.labels)} for r in tr.itertuples()]
    for f in flag_rows:
        f["drop"] = D.drop_decision(f)

flags = pd.DataFrame(flag_rows).set_index("row_id")
tr = tr.join(flags[["drop"]], on="row_id")
flag_table = flags.join(tr.set_index("row_id")[["split", "task_type"]]).groupby(["split", "task_type"]).mean().round(4)
display(flag_table)

results["audit"] = {
    "aligned_rules": aligned,
    "flag_rate_overall": flags.mean().round(4).to_dict(),
    "flag_rate_by_split_task": {f"{s}/{t}": v for (s, t), v in flag_table.to_dict("index").items()},
    "drop_counts": tr.groupby("split")["drop"].sum().astype(int).to_dict(),
    "examples": {},
}
for flag in [c for c in flags.columns if c != "drop"]:
    ids = flags.index[flags[flag]][:2]
    results["audit"]["examples"][flag] = [short(tr.loc[i, "answer"], 200) for i in ids]
print(results["audit"]["drop_counts"])

# %% [markdown]
# ## 4. Input lengths per tokenizer
# Length of `[CLS] prompt [SEP] answer [SEP]` in tokens. This shows how much a 512-token model
# (BERTurk) would have to cut, and confirms 8,192 is enough for the long-context encoders.

# %%
from transformers import AutoTokenizer

TOKENIZERS = {
    "tabibert": "boun-tabilab/TabiBERT",
    "mmbert": "jhu-clsp/mmBERT-base",
    "newmind_modernbert_tr": "newmindai/ModernBERT-tr-uncased-stsb-HD",
    "berturk": "dbmdz/bert-base-turkish-cased",
}
token_stats = {}
for name, model_id in TOKENIZERS.items():
    tok = AutoTokenizer.from_pretrained(model_id)
    n_tokens = np.array([
        len(ids) for ids in tok(list(tr["prompt"]), list(tr["answer"]))["input_ids"]
    ])
    token_stats[name] = {
        "mean": float(n_tokens.mean()), "median": float(np.median(n_tokens)),
        "p90": float(np.percentile(n_tokens, 90)), "p99": float(np.percentile(n_tokens, 99)),
        "max": int(n_tokens.max()),
        "share_over_512": float((n_tokens > 512).mean()),
        "share_over_4096": float((n_tokens > 4096).mean()),
        "share_over_8192": float((n_tokens > 8192).mean()),
    }
    tr[f"tokens_{name}"] = n_tokens
    print(name, {k: round(v, 3) for k, v in token_stats[name].items()})
results["token_lengths"] = token_stats

# %% [markdown]
# ## 5. Grouped splits
# All 6 responses of one source stay in the same split, so near-duplicate answers cannot leak.
#
# - **train / validation**: from the official train split (validation = 5% of sources per task)
# - **dev**: 30 test sources per task (540 responses) — thresholds and model choice for every system
# - **test**: the remaining 360 test sources (2,160 responses) — evaluated once at the end
#
# A "clean" view of each split excludes rows the audit would drop.

# %%
if aligned:
    splits = D.make_splits(tr, group_col="source_id")
    dropped = set(tr.loc[tr["drop"], "row_id"])
    splits_out = {
        **splits,
        **{f"{k}_clean": [i for i in v if i not in dropped] for k, v in splits.items()},
        "meta": {"seed": 482, "dev_sources_per_task": 30, "val_frac": 0.05,
                 "alignment": best_name, "commit": commit},
    }
    with open(f"{DRIVE_DIR}/data/splits.json", "w") as f:
        json.dump(splits_out, f)
    results["splits"] = {k: len(v) for k, v in splits_out.items() if k != "meta"}
    print(results["splits"])
else:
    results["splits"] = None
    print("Alignment not accepted: splits need the fallback grouping (send the results back first).")

# %% [markdown]
# ## Save results
# Send back `MyDrive/hdtr/results/01_audit.json` and `MyDrive/hdtr/data/splits.json`.

# %%
out_path = f"{DRIVE_DIR}/results/01_audit.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
tr.drop(columns=["prompt"]).to_parquet(f"{DRIVE_DIR}/data/ragtruth_tr_audit.parquet")
print("saved", out_path)
