# colab-runtime: cpu
# %% [markdown]
# # 04 · Augmentation: native Turkish training data from DeepSeek
#
# **BIL482 Term Project — Turkish RAG Hallucination Detector**
# İshak Türk · 23120205003
#
# Turns Turkish Wikipedia passages (`alexandrainst/multi-wiki-qa`, `tr`) into RAGTruth-style
# training examples. One DeepSeek call per row returns a **faithful** answer (QA or summary) and,
# for half of the rows, a copy with 1–3 inserted errors marked `<HAL>…</HAL>`. The tags become
# character spans, and every row is validated (Turkish, complete, numbers grounded, tags balanced,
# small edit only, errors not copied from the context).
#
# **Two modes:**
# 1. `MODE = "gate"`: 30 rows (20 QA + 10 summary, half with errors), about **$0.03**. You review
#    them by eye; we continue only if ≥ 85% are acceptable.
# 2. `MODE = "full"`: all ~3,800 rows, about **$2.70 at peak price, half that off-peak**. Run it only
#    after the gate passes, and preferably off-peak: weekdays outside **04:00–07:00 and 09:00–13:00
#    Turkish time**, or any time at the weekend. A $4 budget cap stops it early if anything goes wrong.
#
# **Runtime:** CPU is enough (no GPU units used). **Secrets:** `GH_TOKEN`, `DEEPSEEK_API_KEY`.
# **Output:** gate → `MyDrive/hdtr/results/04_gate.json` · full → `MyDrive/hdtr/results/04_augmentation.json`.
# Rerunning resumes: finished rows are kept in a JSONL file on Drive and never paid for twice.

# %%
MODE = "gate"          # "gate" or "full"
GATE_PASSED = False    # set to True only after the gate review passed; required for MODE = "full"

# %% [markdown]
# ## Setup

# %%
import base64, os, subprocess, sys

REPO = "ishakturk/Turkish-RAG-Hallucination-Detector"
REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
DRIVE_DIR = "/content/drive/MyDrive/hdtr"

from google.colab import drive, userdata
drive.mount("/content/drive")
for sub in ("results", "cache", "augment"):
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
import html, json, random, time
import pandas as pd
from IPython.display import HTML, display
from openai import AsyncOpenAI

from hdtr import augment as A

if MODE not in ("gate", "full"):
    raise ValueError("MODE must be 'gate' or 'full'")
if MODE == "full" and not GATE_PASSED:
    raise RuntimeError("Full mode needs GATE_PASSED = True (after the 30-example review passed).")

# %% [markdown]
# ## Source data and the title split
# Split by Wikipedia article: ~4,000 rows for augmentation, ~1,000 reserved as contexts for the
# native test set. The split is fixed in the repo (`data/mwqa_split.json`), and this cell checks
# that it reproduces.

# %%
mwqa = A.load_mwqa_tr(cache=f"{DRIVE_DIR}/cache/mwqa_tr.parquet")
split = A.split_titles(mwqa)
with open(f"{REPO_DIR}/data/mwqa_split.json", encoding="utf-8") as f:
    committed = json.load(f)
assert split["augment_titles"] == committed["augment_titles"], "title split differs from the committed one"
aug_rows = mwqa[mwqa["title"].isin(split["augment_titles"])]
print(f"rows: {len(mwqa)} | augmentation: {len(aug_rows)} rows / {len(split['augment_titles'])} articles | "
      f"native pool: {len(mwqa) - len(aug_rows)} rows / {len(split['native_titles'])} articles")

# %% [markdown]
# ## Plans
# Each plan fixes the task (QA or summary), the RAGTruth-TR prompt, and whether and how to insert
# errors. Rows whose answer sits in a list or category section are skipped.

# %%
if MODE == "gate":
    sample = aug_rows.sample(60, random_state=482)
    pattern = [("QA", True), ("QA", False), ("Summary", True), ("QA", True), ("QA", False), ("Summary", False)]
    plans = A.make_plans(sample, seed=482, forced=pattern)[:30]
    out_jsonl, budget, concurrency = f"{DRIVE_DIR}/augment/gate_v{A.PROMPT_VERSION}.jsonl", 0.5, 10
else:
    plans = A.make_plans(aug_rows, seed=482)
    out_jsonl, budget, concurrency = f"{DRIVE_DIR}/augment/full_v{A.PROMPT_VERSION}.jsonl", 4.0, 20

print(f"{len(plans)} plans | QA {sum(p['task_type'] == 'QA' for p in plans)} | "
      f"Summary {sum(p['task_type'] == 'Summary' for p in plans)} | "
      f"with errors {sum(p['spec']['hallucinate'] for p in plans)}")
print("Peak price right now." if A.is_peak() else "Off-peak price right now (half price).")

# %% [markdown]
# ## Generate
# Resumable: rows already in the JSONL on Drive are skipped. The file name carries the prompt version,
# so a changed prompt always starts a fresh file. Each row gets up to 2 attempts if
# validation fails.

# %%
client = AsyncOpenAI(api_key=userdata.get("DEEPSEEK_API_KEY"), base_url=A.BASE_URL)
t0 = time.time()
run = await A.run_batch(client, plans, out_jsonl, concurrency=concurrency, budget_usd=budget)
print(run, f"| {time.time() - t0:.0f} s")
if run["stopped_for_budget"]:
    print("STOPPED: the budget cap was reached. Send the results file before running again.")

lines = A.read_jsonl(out_jsonl)
plan_ids = {p["id"] for p in plans}
lines = [l for l in lines if l["id"] in plan_ids]
stats = A.batch_stats(lines)
print(json.dumps(stats, indent=2, ensure_ascii=False))

# %% [markdown]
# ## Review (gate mode)
# Read every card. A row is **acceptable** when:
# - the faithful answer is fully supported by the context and reads like natural Turkish;
# - in the version with errors, every highlighted part really is wrong or unsupported by the context,
#   the highlight covers the error and not much more, and the rest of the text is unchanged.
#
# A CSV with one line per row is also saved to `MyDrive/hdtr/augment/gate_review.csv`. Open it with
# Google Sheets, put `1` (ok) or `0` (not ok) in the two verdict columns, add a note where useful, then
# download it as CSV and send it back with `04_gate.json`.

# %%
plan_by_id = {p["id"]: p for p in plans}

def esc(s):
    return html.escape(s).replace("\n", "<br>")

def marked_html(text, spans):
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += [esc(text[pos:s["start"]]), "<mark>", esc(text[s["start"]:s["end"]]), "</mark>"]
        pos = s["end"]
    return "".join(out + [esc(text[pos:])])

def card(line):
    plan = plan_by_id[line["id"]]
    head = f"<b>{line['id']}</b> · {plan['task_type']} · {esc(plan['title'])} · spec {esc(json.dumps(plan['spec'], ensure_ascii=False))}"
    body = f"<details><summary>prompt (click)</summary><div style='font-size:12px'>{esc(plan['prompt'])}</div></details>"
    if not line["accepted"]:
        return (f"<div style='border:2px solid #c33;padding:8px;margin:8px 0'>{head}<br><b>REJECTED</b> "
                f"{esc(str(line['reasons']))}{body}<pre style='white-space:pre-wrap'>{esc(line['raw'][:1500])}</pre></div>")
    recs = {r["variant"]: r for r in line["records"]}
    parts = [f"<p><b>faithful</b><br>{esc(recs['faithful']['answer'])}</p>"]
    if "hallucinated" in recs:
        h = recs["hallucinated"]
        errs = "; ".join(f"{e.get('type')}: '{e.get('original', '')}' → '{e.get('changed', '')}'" for e in h["errors"])
        parts.append(f"<p><b>with errors</b><br>{marked_html(h['answer'], h['labels'])}</p><p><i>{esc(errs)}</i></p>")
    return f"<div style='border:1px solid #999;padding:8px;margin:8px 0'>{head}{body}{''.join(parts)}</div>"

if MODE == "gate":
    display(HTML("<style>mark{background:#ffb3b3}</style>" + "".join(card(l) for l in lines)))
    review = []
    for l in lines:
        plan = plan_by_id[l["id"]]
        recs = {r["variant"]: r for r in l["records"]}
        h = recs.get("hallucinated")
        review.append({
            "id": l["id"], "task": plan["task_type"], "title": plan["title"],
            "question": mwqa.loc[plan["row"], "question"] if plan["task_type"] == "QA" else "",
            "accepted_by_checks": l["accepted"], "rejection_reasons": "" if l["accepted"] else json.dumps(l["reasons"]),
            "faithful": recs["faithful"]["answer"] if "faithful" in recs else "",
            "with_errors_marked": A.mark_spans(h["answer"], h["labels"]) if h else "",
            "errors": json.dumps(h["errors"], ensure_ascii=False) if h else "",
            "verdict_faithful": "", "verdict_with_errors": "", "note": "",
        })
    pd.DataFrame(review).to_csv(f"{DRIVE_DIR}/augment/gate_review.csv", index=False, encoding="utf-8-sig")
    print("saved", f"{DRIVE_DIR}/augment/gate_review.csv")

# %% [markdown]
# ## Build the augmentation set (full mode)

# %%
if MODE == "full":
    frame = A.records_frame(lines)
    frame["errors"] = frame["errors"].map(lambda e: json.dumps(e, ensure_ascii=False))
    frame.to_parquet(f"{DRIVE_DIR}/augment/mwqa_aug.parquet")
    print(f"saved {len(frame)} records to {DRIVE_DIR}/augment/mwqa_aug.parquet")
    display(frame.groupby(["task_type", "variant"]).size().rename("records").to_frame())

# %% [markdown]
# ## Save results
# Send back the file printed below (and, after the gate, the filled-in `gate_review.csv`).

# %%
results = {"notebook": "04_augmentation", "mode": MODE, "commit": commit,
           "time": time.strftime("%Y-%m-%d %H:%M"), "model": A.MODEL, "prompt_version": A.PROMPT_VERSION, "run": run, "stats": stats,
           "split": {"augment_rows": len(aug_rows), "augment_titles": len(split["augment_titles"]),
                     "native_titles": len(split["native_titles"])},
           "plans": len(plans)}
if MODE == "gate":
    keep = ("id", "task_type", "spec", "accepted", "records", "reasons", "attempts", "cost_usd")
    results["lines"] = [{**{k: l[k] for k in keep}, "prompt": plan_by_id[l["id"]]["prompt"],
                         "raw": None if l["accepted"] else l["raw"]} for l in lines]
    out_path = f"{DRIVE_DIR}/results/04_gate.json"
else:
    accepted = [l for l in lines if l["accepted"]]
    random.Random(0).shuffle(accepted)
    results["examples"] = [{"id": l["id"], "task_type": l["task_type"],
                            "records": [{"variant": r["variant"], "answer_marked": A.mark_spans(r["answer"], r["labels"])}
                                        for r in l["records"]]} for l in accepted[:12]]
    out_path = f"{DRIVE_DIR}/results/04_augmentation.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print("saved", out_path)
