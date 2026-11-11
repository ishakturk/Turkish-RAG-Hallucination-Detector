# colab-runtime: t4
# %% [markdown]
# # Turkish RAG Hallucination Detector
#
# **BIL482 Natural Language Processing — Term Project, İstanbul Medeniyet University, Fall 2026**
# İshak Türk · 23120205003 · [GitHub repository](https://github.com/ishakturk/Turkish-RAG-Hallucination-Detector)
#
# A RAG system answers a question from retrieved passages. An answer **hallucinates** when it states
# something the passages do not support, even if it happens to be true in the real world. This project
# trains a small encoder that marks those unsupported parts of a Turkish answer, compares it with every
# existing Turkish detector under one protocol, and tests all of them on a new set of natural Turkish answers.
#
# **What this notebook does**
#
# | Mode | Runtime | Steps |
# |---|---|---|
# | `quick` (default) | free Colab T4, ~10 min | 1 · data overview → 2 · all tables and figures from the saved results → 3 · native Turkish results **recomputed** from saved predictions → 4 · the released model, live on examples and on the whole native set |
# | `full` | L4/A100, ~2 h | everything in `quick`, then 5 · trains the final model (R2, seed 42) from scratch and evaluates it and the classical baselines on the final test |
#
# The numbered notebooks in `notebooks/` (00–11) are the ones that produced the saved results, each run
# once in Colab; this notebook reads their outputs from `results/` and checks them.
# **Run:** Runtime → Run all. No secrets or Drive access are needed.

# %%
MODE = "quick"          # "quick" or "full"
MODEL_REPO = "ishakturk/turkish-rag-hallucination-detector"   # the released final model (notebook 11)

# %% [markdown]
# ## 0 · Setup
# Clones the repository (in Colab) and installs the project package `hdtr`.

# %%
import os, subprocess, sys

REPO_URL = "https://github.com/ishakturk/Turkish-RAG-Hallucination-Detector.git"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_DIR = "/content/Turkish-RAG-Hallucination-Detector"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[colab]"], check=True)
else:   # running from a local checkout
    REPO_DIR = os.environ.get("HDTR_REPO_DIR") or os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebooks" else ".")
sys.path.insert(0, f"{REPO_DIR}/src")
OUT_DIR = os.path.join(REPO_DIR, "main_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

import json
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

from hdtr import figures as F
from hdtr import reproduce as R
from hdtr.postprocess import char_probs_to_spans

pd.set_option("display.max_colwidth", 80)

def load_result(name):
    with open(f"{REPO_DIR}/results/{name}", encoding="utf-8") as f:
        return json.load(f)

train_res, frozen, final_res, native_res = (load_result(n) for n in
    ("06_train.json", "07_thresholds.json", "08_final_test.json", "10_native_test.json"))
systems = frozen["systems"]
print("repository:", REPO_DIR, "| mode:", MODE)

# %% [markdown]
# ## 1 · Data
#
# **RAGTruth-TR** is the machine-translated Turkish version of RAGTruth: 17,790 answers written by six
# LLMs for 2,965 sources (question answering, data-to-text, summarization), with hallucinated character
# spans. We recovered which source each answer belongs to (by aligning with the English original) and split
# **by source**, so no context appears in two splits. The thresholds and every model choice use the
# **dev** split; the **final test** was scored once.
#
# The **native Turkish test set** is new: Turkish Wikipedia and Turkish law passages, questions partly
# rewritten so that the passage answers only part of them, and answers written directly in Turkish by
# Qwen3-8B and DeepSeek. Labels come from two LLM label passes, adjudicated and checked by a human on 50 rows.

# %%
with open(f"{REPO_DIR}/data/splits.json") as f:
    splits = json.load(f)
display(pd.DataFrame({"answers": {k: len(v) for k, v in splits.items() if isinstance(v, list)}}).T)

native_path, preds_path = f"{REPO_DIR}/{R.NATIVE_FILE}", f"{REPO_DIR}/{R.NATIVE_PREDICTIONS}"
native = R.load_native_test(native_path) if os.path.exists(native_path) else None
if native is None:
    print("data/native_test.jsonl is not in this checkout yet (it is added by notebook 11).")
else:
    counts = native.groupby(["domain", "generator", "question_type"])["label"].agg(hallucinated="sum", answers="count")
    display(counts)
    print(f"{len(native)} answers, {native['label'].sum()} hallucinated ({native['label'].mean():.0%}), "
          f"{sum(len(s) for s in native['spans'])} gold spans")

# %% [markdown]
# ### How an answer is scored
# The model gives every character of the answer a probability of being unsupported. A word is flagged when
# its highest character probability reaches the threshold τ; flagged words separated only by a short word
# merge into one span. Turkish suffixes stay with their word: below, only the suffix *-a* of *"Ocak'a"* gets
# a high probability, yet the whole word is flagged, never the suffix alone. The probabilities are made up
# to show the rule.

# %%
answer = "Fuar 3 Ocak'a kadar açık kalacak ve 2019'da kuruldu."
probs = np.zeros(len(answer), dtype=np.float32)
for word, p in (("2019'da", 0.92), ("kuruldu", 0.81), ("Ocak'a", 0.3)):
    k = answer.index(word)
    probs[k:k + len(word)] = p
probs[answer.index("'a") + 1] = 0.6          # only the suffix character is high
for tau in (0.5, 0.7):
    spans = char_probs_to_spans(answer, probs, tau)
    print(f"τ = {tau}:", [s["text"] for s in spans])
    display(HTML(R.highlight_html(answer, spans)))

# %% [markdown]
# ## 2 · Results on the translated final test
# Everything below is rebuilt from the committed result files of notebooks 06–10 (no model is run).
# Our runs are averaged over three seeds (± standard deviation); brackets are 95% bootstrap confidence
# intervals that resample sources.

# %%
paths = F.build_all(train_res, frozen, final_res, f"{OUT_DIR}/figures", native=native_res, lang="en")
tables = f"{OUT_DIR}/figures"
display(Markdown("**Table 1 · Final test (2,160 answers), frozen dev thresholds**"))
display(pd.read_csv(f"{tables}/table1_main.csv").set_index("System"))
display(Image(f"{tables}/fig1_main_results.png"))

# %% [markdown]
# **Pre-registered comparisons.** Q1 (final model vs the best existing detector chosen on dev) is the one
# confirmatory test: the model counts as better only if the whole 95% interval is above zero.
# Q2–Q5 each change one thing (base model, hallucination pretraining, training recipe, augmentation).

# %%
display(pd.read_csv(f"{tables}/table3_q1_q5.csv"))
display(Image(f"{tables}/fig2_q1_q5_forest.png"))
display(pd.DataFrame(final_res["q1"]["verdict"]).T)

# %%
display(Markdown("**Span F1 per task** (summarization is the hardest task for every system)"))
per_task = pd.read_csv(f"{tables}/table2_per_task.csv")
display(per_task.set_index(per_task.columns[0]))
display(Image(f"{tables}/fig3_span_f1_per_task.png"))
display(Image(f"{tables}/fig4_threshold_curves.png"))
display(Image(f"{tables}/fig5_training_curves.png"))

# %% [markdown]
# ## 3 · Native Turkish test set, recomputed from saved predictions
# `data/native_predictions.npz` holds the character probabilities that every run produced on the 184
# native answers (notebook 10). Scoring them again with the frozen thresholds must give exactly the numbers
# saved in `results/10_native_test.json`.

# %%
if native is None or not os.path.exists(preds_path):
    print("Native files are not in this checkout yet; showing the saved table only.")
else:
    native_probs = R.load_native_predictions(preds_path, native)
    rescored = R.rescore(native, native_probs, systems)
    check = R.compare_with_saved(rescored, native_res)
    print(f"{len(native_probs)} runs rescored | largest difference from the saved results: {check['abs_diff'].max():.2e}")
    assert check["abs_diff"].max() < 1e-9
display(Markdown("**Native test (184 answers, 47 hallucinated)**: descriptive, CIs resample contexts"))
display(pd.read_csv(f"{tables}/table4_native.csv").set_index("System"))
display(Image(f"{tables}/fig7_native_rates.png"))
display(Image(f"{tables}/fig6_translated_vs_native.png"))

# %% [markdown]
# **Reading the native results.** Every trained detector loses a lot on natural Turkish answers
# (final model: example F1 0.766 → 0.458). It is still precise (what it flags is usually wrong) but finds
# only a third of the hallucinations, which here are short and subtle: a year taken from a nearby date, a
# true fact that the passage does not contain. Training with natural Turkish data (R4) narrows the gap most.
# Almost all hallucinations come from Qwen3-8B on questions the passage answers only in part, so the set
# measures that kind of error best.

# %% [markdown]
# ## 4 · The released model, live
# Downloads the final model (R2, mmBERT-base, run `r2_mmbert_s42`) from Hugging Face and runs it on a few
# native examples (red: predicted, underlined: gold), then on the whole native set.

# %%
model = tok = None
if MODEL_REPO is None:
    print("MODEL_REPO is not set yet; skipping the live part.")
else:
    import torch
    from hdtr.infer import load_detector, predict_many
    on_gpu = torch.cuda.is_available()
    native_bf16 = on_gpu and torch.cuda.get_device_capability()[0] >= 8      # T4 has no native bf16
    tau = systems[R.RELEASE_RUN]
    hf_token = None
    if IN_COLAB:   # optional: only needed while the model repository is still private
        from google.colab import userdata
        try:
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    try:
        tok, model = load_detector(MODEL_REPO, dtype=torch.bfloat16 if native_bf16 else torch.float32,
                                   **({"token": hf_token} if hf_token else {}))
        print("loaded", MODEL_REPO, "| thresholds:", {k: tau[k] for k in ("tau_example", "tau_span")},
              "| device:", next(model.parameters()).device)
    except Exception as exc:   # e.g. no network, or the repository is still private
        print(f"Could not download {MODEL_REPO} ({type(exc).__name__}); skipping the live part.")

# %%
EXAMPLES = ["wiki-01711", "wiki-03395", "wiki-00490", "legal-0082", "legal-0190"]
if model is not None and native is not None:
    for i in EXAMPLES:
        r = native.loc[i]
        out = R.detect(model, tok, r["question"], r["passages"], r["answer"], tau["tau_span"], prompt=r["prompt"])
        display(HTML(f"<p><b>{i}</b> · {r['generator']} · {r['question_type']}<br><i>{r['question']}</i><br>"
                     f"{R.highlight_html(r['answer'], out['spans'], r['spans'])}<br>"
                     f"<small>max probability {out['max_prob']:.2f}</small></p>"))

# %% [markdown]
# **Whole native set with the downloaded model.** Same thresholds, same scoring as notebook 10. On a GPU
# with bf16 the numbers should match the saved run closely; in float32 (T4, CPU) they can move slightly.

# %%
if model is not None and native is not None:
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=native_bf16):
        live_probs, timing = predict_many(model, tok, zip(native["prompt"], native["answer"]), max_length=4096)
    live = R.rescore(native, {R.RELEASE_RUN: live_probs}, systems)["per_run"][R.RELEASE_RUN]["tuned"]["overall"]
    saved = native_res["per_run"][R.RELEASE_RUN]["tuned"]["overall"]
    display(pd.DataFrame({"live": {m: live[m] for m in ("example_f1", "example_auroc", "span_f1")},
                          "saved (notebook 10)": {m: saved[m] for m in ("example_f1", "example_auroc", "span_f1")}}).round(3))
    print(f"{timing['sec_per_example'] * 1000:.0f} ms per answer")

# %% [markdown]
# ## 5 · Full mode: train the final model again and evaluate
# Only when `MODE = "full"` (L4 or A100, ~2 h). Downloads RAGTruth-TR and the English RAGTruth (to recover
# source ids), trains R2 with seed 42 from `configs/r2.yaml`, tunes its thresholds on dev and scores the
# final test once; the classical baselines are refit as in notebook 02. GPU training is not bit-exact, so
# the new numbers are compared with the saved run rather than required to equal it.

# %%
if MODE == "full":
    import dataclasses, torch
    from hdtr import data as D
    from hdtr.baselines import LexicalOverlap, WordLogReg
    from hdtr.evaluation import score, tune_thresholds
    from hdtr.infer import predict_many
    from hdtr.train import RunConfig, train

    tr = D.load_ragtruth_tr()
    signals = D.align_by_order(tr, D.load_ragtruth_en(cache_dir=f"{OUT_DIR}/cache"))
    assert signals["pass"].mean() > 0.99
    tr["source_id"] = signals["source_id"].values
    dev, test = tr.loc[splits["dev"]], tr.loc[splits["test"]]

    cfg = RunConfig.from_yaml(f"{REPO_DIR}/configs/r2.yaml")
    cfg = dataclasses.replace(cfg, name="r2_mmbert_s42_rerun", seed=42)
    run = train(cfg, tr.loc[splits["train_clean"]], tr.loc[splits["validation_clean"]], f"{OUT_DIR}/r2_rerun")
    model_full, tok_full = run["model"].eval(), run["tokenizer"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dev_probs, _ = predict_many(model_full, tok_full, zip(dev["prompt"], dev["answer"]), max_length=cfg.max_length)
        test_probs, _ = predict_many(model_full, tok_full, zip(test["prompt"], test["answer"]), max_length=cfg.max_length)
    th = tune_thresholds(dev, dev_probs)
    new = {"example_f1": score(test, test_probs, th["tau_example"])["overall"]["example_f1"],
           "span_f1": score(test, test_probs, th["tau_span"])["overall"]["span_f1"]}
    old = final_res["per_run"]["r2_mmbert_s42"]["full"]["tuned"]["overall"]
    display(pd.DataFrame({"retrained": new, "saved r2_mmbert_s42": {m: old[m] for m in new}}).round(3))
    print("dev thresholds:", {k: float(th[k]) for k in ("tau_example", "tau_span")})

    train_clean = tr.loc[splits["train_clean"]]
    lex = LexicalOverlap().fit(train_clean["prompt"].sample(3000, random_state=0), train_clean["answer"].sample(3000, random_state=0))
    fit = train_clean.sample(4000, random_state=0)
    logreg = WordLogReg(lex).fit(fit["prompt"], fit["answer"], fit["labels"])
    rows = {}
    for name, m in {"classical_lexical_overlap": lex, "classical_word_logreg": logreg}.items():
        s = systems[name]
        p = [m.predict_char_probs(a, b) for a, b in zip(test["prompt"], test["answer"])]
        rows[name] = {"example_f1": score(test, p, s["tau_example"])["overall"]["example_f1"],
                      "saved": final_res["per_run"][name]["full"]["tuned"]["overall"]["example_f1"]}
    display(pd.DataFrame(rows).T.round(3))
else:
    print("MODE is 'quick': training skipped.")

# %% [markdown]
# ## Summary
# - Under one protocol, the final model (multilingual mmBERT-base fine-tuned on cleaned RAGTruth-TR) beats the
#   best existing Turkish detector chosen on dev by **+0.070 example F1** and **+0.100 span F1** (95% CIs above
#   zero), and every existing detector in exploratory comparisons.
# - Most of the gain comes from the base model; thresholds and evaluation rules move results a lot, which
#   is why every decision was fixed before the test was opened.
# - On natural Turkish answers every detector drops sharply; training with natural Turkish data helps most.
#
# Report (Turkish), article (English), code and data: see the repository.
