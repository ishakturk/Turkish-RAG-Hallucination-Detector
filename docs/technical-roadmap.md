# Technical Roadmap: Turkish RAG Hallucination Detector

- **Course:** BIL482 term project. İshak Türk, 23120205003
- **Deadline:** Sunday 7 December 2026, 23:00 (Google Classroom). Presentations on 11, 18 or 25 December.
- **Written:** 2026-11-06. Replaces the method and timeline sections (7–13) of the original `roadmap.md`. Its sections 1–6 (brief, rubric, decision history, problem statement) still apply.
- **Companion docs:** [`verification-2026-11-06.md`](verification-2026-11-06.md) (fact checks, IEEE references) and [`model-and-data-research-2026-11-06.md`](model-and-data-research-2026-11-06.md) (model and data survey).

---

## 0. Final framing and decisions

**One sentence:** *We train Turkish hallucination detectors on a Turkish-native long-context encoder, compare them with every existing Turkish detector under one reproducible protocol, and measure, for the first time, how they perform on natural Turkish rather than translated text.*

**Contributions (in report order):**
1. A public, reproducible evaluation of all Turkish detectors at example, word and span level, documenting the reporting inconsistencies in prior work.
2. A native Turkish test set (~200 examples), hand-labeled, with human–LLM agreement reported.
3. Our detector (TabiBERT-based), plus controlled answers to two questions: Turkish-native vs multilingual base, and whether multilingual hallucination training transfers to Turkish.

**Decision log:**

| # | Decision |
|---|---|
| D1 | Official test (2,700 responses = 450 sources × 6) is split once, grouped by source: **dev-from-test** = 90 sources / 540 responses, **final test** = 360 sources / 2,160 responses, stratified by task. All systems use the same dev for thresholds and checkpoint choice. The final test is run **once**. |
| D2 | Primary metrics: **example-level F1** and **span-level F1** for the hallucinated class on the final test. Secondary: word-level F1, macro F1, AUROC from probabilities. Meaningful gain: the 95% bootstrap CI of the difference vs the best baseline excludes 0. Written into `analysis-plan.md` before any final-test run. |
| D3 | Baselines: 6 Newmind detectors, KR Labs v2 mmBERT (zero-shot), 2 classical models. Optional: KR Labs v2 Qwen-2B. |
| D4 | Runs: R0 TabiBERT with old-style recipe; R1 TabiBERT with our recipe; R2 mmBERT-base with our recipe; R3 KR Labs v2 mmBERT with our recipe; R4 = best of R1–R3 plus augmentation. |
| D5 | Data audit uses alignment to the original English RAGTruth. The same audit runs on test, and we report both full test and clean test. |
| D6 | Augmentation: ~4,000 native Turkish examples generated in one batch with **DeepSeek `deepseek-flash`** (thinking off) from `alexandrainst/multi-wiki-qa` (`tr`). Estimated cost $2.50–5 (see §2.6). Groq / Qwen3.8-27B is the fallback. |
| D7 | Native test set: ~200 examples. The user labels at example level, blind. Claude labels independently; we compute κ, the user adjudicates, and spans are LLM-proposed and user-edited. Never used for training. |
| D8 | Workflow: private GitHub repo, made public at submission. Notebooks are run manually in Colab, and each one ends with a `results/*.json`. |
| D9 | Hugging Face release after the project. For submission, the trained model is downloadable without login (decided in Phase 4). |

---

## 1. Repository and Colab workflow

### 1.1 Repository layout

```
Turkish RAG Hallucination Detector/
  README.md
  pyproject.toml              # package "hdtr", pinned deps
  docs/                       # roadmap, verification, research, analysis-plan, labeling-guide, report drafts
  src/hdtr/
    data.py                   # load RAGTruth-TR, align to English, audit, group ids, splits
    labels.py                 # char spans <-> token labels via offset_mapping
    train.py                  # Trainer setup, weighted / focal loss, callbacks
    infer.py                  # char-level probabilities for any token-classification model
    postprocess.py            # word snapping, span merging, thresholds
    metrics.py                # example / word / span metrics, AUROC, bootstrap, permutation test
    baselines.py              # classical baselines
    augment.py                # DeepSeek generation, <HAL> parsing, validation
    native.py                 # native test set build, label import, kappa
    viz.py                    # all figures, one style
  configs/                    # one YAML per run (r0.yaml ... r4.yaml, pilot_*.yaml)
  notebooks/                  # 00-09 working notebooks + main.ipynb (deliverable)
  results/                    # small JSON/CSV summaries committed to git
  data/                       # small committed data: split id lists, native test set, labeling CSVs
  tests/                      # pytest: label alignment, Turkish casing, metrics
```

Large files (checkpoints, predictions, augmented data) go to Google Drive under `MyDrive/hdtr/`. Only small summaries go into git.

### 1.2 How each notebook runs in Colab

1. Mount Drive.
2. Clone the private repo using a fine-grained, read-only GitHub token stored in **Colab Secrets** as `GH_TOKEN`.
3. `pip install -e .`
4. Log the GPU, library versions and git commit hash.
5. Run.
6. Write outputs to Drive, and a summary to `results/<nb>_<date>.json`.

**What you send back:** the `results/*.json` file, plus any figure the notebook names. Download them from Drive and drop them into the project's `results/` folder. I read and commit them. You never need to send whole notebooks.

**Secrets:** `DEEPSEEK_API_KEY` and `GH_TOKEN` live only in Colab Secrets, read with `google.colab.userdata.get(...)`. They never go in a cell, the repo, or chat.

### 1.3 Pinned environment
- `transformers` 4.x (`<5`; the custom code of EuroBERT and gte needs it), `datasets`, `accelerate`, `lettucedetect==0.2.3` (used only to load baselines, cross-checked against our metrics), `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `openai` (DeepSeek's API is OpenAI-compatible), `vllm` (only in the native-answer notebook), `pytest`.
- Exact versions are fixed in Phase 0 after a smoke test.
- Precision: `bf16` on L4 and A100. T4 is used only for quick mode (fp16 or fp32 inference).

---

## 2. Technical specifications

### 2.1 Input format and labels
- **Input:** `[CLS] prompt [SEP] answer [SEP]`. RAGTruth-TR's `prompt` already contains the instruction, context and question.
- **Labels:**
  - Prompt tokens get `-100`.
  - Answer tokens get `1` if their character range overlaps any gold span, otherwise `0`.
  - Alignment uses the fast tokenizer's `offset_mapping` on the **original text**. **We never call `.lower()` ourselves** (Turkish `İ`/`I` problem).
- **Truncation:** `max_length=8192`, `truncation="only_first"`. Only the prompt is ever cut, never the answer. The number of truncated examples is logged per tokenizer.
- **Unit tests:** strings with `İstanbul'a`, `IŞIK`, `Ocak'a`, curly `’`, and spans at the answer's start and end.

### 2.2 Comparable metrics across different tokenizers
Token-level scores from different tokenizers cannot be compared. Every model therefore outputs **per-character probabilities** of the answer (each token's probability spread over its characters), and all metrics are computed from those:

| Level | Definition |
|---|---|
| Example | Score = max character probability. Predicted hallucinated if any predicted span exists after the threshold. Gold = answer has ≥1 span. |
| Word | Answer split into words (whitespace plus punctuation, apostrophe suffix kept with its word). Word probability = max over its characters. P/R/F1 for the hallucinated class, plus macro F1, plus AUROC. |
| Span | Character-overlap precision and recall, as in LettuceDetect (overlapping characters / predicted characters, and / gold characters), plus F1. |

Reported per task (QA, Data2txt, Summary) and overall. We also run the `lettucedetect` evaluator for one model as a sanity check: on the same predictions, its numbers should match ours.

**Significance:**
- Bootstrap: resample **sources**, not responses, with 2,000 replicates. Report 95% CIs for every primary metric and for the difference between systems.
- Paired permutation test (10,000 permutations) for our best model vs the best baseline.

### 2.3 Training recipe ("our recipe")
| Setting | Value (updated after the 03 pilot, 9 Nov) |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| LR | 3e-5 for every run (best in the pilot), 6% warmup, linear decay |
| Epochs | up to 4, evaluated every ½ epoch, early stopping with patience 2 on validation word-F1 |
| Loss | class-weighted cross-entropy (weight for class 1 = √(neg/pos)); focal loss was worse in the pilot and is dropped |
| Batch | effective 16 (per-device batch × gradient accumulation), length-grouped sampling, gradient checkpointing if out of memory |
| Data | cleaned RAGTruth-TR train (minus validation) |
| Validation | 5% of train sources, grouped (for early stopping) |
| Model choice | best checkpoint by validation; final comparison and thresholds on dev-from-test |
| Checkpointing | to Drive every 500 steps plus best; `resume_from_checkpoint` after disconnects |
| Seeds | 3 seeds (42, 1234, 2026) for R1–R4: training is cheap and pilot differences are within single-seed noise |
| Tokenization | default tokenizer output; word-split input for TabiBERT gave no gain in the pilot and is dropped |

**R0 ("Newmind-style") recipe:** 6 fixed epochs, lr 1e-5, batch 4, plain cross-entropy, no validation (last checkpoint used), uncleaned data. If compute is tight, R0 is capped at the same number of optimizer steps as R1 and this is documented.

Note for R3: its starting checkpoint saw the *English* RAGTruth train set, so its validation set (carved from train) is mildly contaminated. Dev-from-test is clean for all runs, and selection happens there.

### 2.4 Post-processing
1. **Word snapping:** a word is flagged if its max character probability ≥ τ. The apostrophe suffix stays with its stem.
2. **Merging:** flagged words separated by ≤1 unflagged short word (≤3 characters) or by punctuation are merged into one span.
3. **Minimum span:** drop spans shorter than 2 characters.
4. **Thresholds** (all tuned on dev-from-test, per model, so every system gets the same treatment): τ_span maximizes span F1, and τ_example maximizes example F1.

Reported as a separate ablation row: raw model vs post-processed.

### 2.5 Data alignment and audit (D5)
- **Alignment to English:**
  - Load the original RAGTruth (English) with `source_id`, either the official RAGTruth release (`response.jsonl` + `source_info.jsonl`) or LettuceDetect's preprocessed version.
  - Hypothesis: RAGTruth-TR was translated in the same row order within each `(split, task_type)`. Check it with three signals: span count, digits in the answer, and length ratio TR/EN within 0.6–1.8.
  - Accept the alignment if ≥95% of rows pass.
- **Fallback grouping:** TF-IDF on the prompt (character 4-grams), agglomerative clustering into groups of about 6 within `(split, task_type)`. Check on 30 random groups by eye.
- **Audit rules** (each counted and reported). A row is flagged if:
  1. any span is outside the answer's bounds or empty;
  2. a span starts or ends mid-word (soft flag: fixed by snapping, not dropped);
  3. the TR/EN span count differs (after alignment);
  4. the TR/EN answer-length ratio is outside 0.5–2.0;
  5. more than 15% of answer tokens are untranslated English (stopword-list test);
  6. there are repeated n-grams (same 4-gram ≥4 times), a sign of degenerate translation;
  7. the answer is truncated (ends mid-sentence while the English answer doesn't).
- **Cleaning policy:** rules 1, 3, 6 and 7 are hard drops. Rules 4 and 5 drop only when combined with another flag. Rule 2 is fixed by snapping. The rules are frozen before any model is trained on the cleaned data.
- **Clean test:** the same rules applied to the final test. We report full test and clean test.

### 2.6 Augmentation generator (D6)

**Generator (decided 2026-11-09): DeepSeek `deepseek-flash`**, called through the OpenAI-compatible API
(`base_url=https://api.deepseek.com`, key in Colab Secrets as `DEEPSEEK_API_KEY`, $5 prepaid).
- **Terms:** the Open Platform ToS (effective 2026-04-29, §4.2(3)) explicitly permits using outputs for "training other models (such as model distillation)".
- **Thinking mode is on by default.** Send `extra_body={"thinking": {"type": "disabled"}}`. Notebook 00 asserts that no reasoning content comes back.
- **JSON mode:** `response_format={"type": "json_object"}`, the word "json" in the prompt plus an example, a sensible `max_tokens`, and a retry on blank content.
- **Limits:** concurrency only (2,500 per account), no RPM/TPM caps. 4,000 calls at ~20 concurrent should take ~15–30 min.
- **Cost estimate** (measured: Turkish passages ≈ 2.5 characters per DeepSeek token, ~1,000 tokens per passage after capping):
  - Per call: ~600 tokens of fixed instructions (billed at the cache-hit price), ~1,050 tokens of passage + question (cache miss), ~500 output tokens.
  - That is ≈ $0.0009 per example at peak and ≈ $0.00046 off-peak.
  - 4,000 examples + 30% retries ≈ **$4.80 peak / $2.40 off-peak**. Native test answers (~200) ≈ $0.15. `deepseek-v4-pro` would be ~3.5× more; use it only if Flash fails the quality gate.
- **Off-peak** (half price): everything except weekdays 01:00–04:00 and 06:00–10:00 UTC (04:00–07:00 and 09:00–13:00 Turkish time).
- **Data note:** multi-wiki-qa `tr` has 5,006 QA rows over only **673 Wikipedia articles**, so the augmentation/native-test split is by article title.

**Fallback (kept for reference): Groq `qwen/qwen3.8-27b`**, called through the OpenAI-compatible API
(`base_url=https://api.groq.com/openai/v1`, key in Colab Secrets as `GROQ_API_KEY`). It replaces
DeepSeek in this section.
- **Licensing:** the model is Apache-2.0. Groq's Services Agreement has no clause against training on outputs (§6.3(e) only bars competing with Groq's cloud). Verified 2026-11-08.
- **Reasoning:** must be switched off (`reasoning_effort="none"`), otherwise thinking tokens eat the budget. Groq's docs disagree on what `"default"` does, so the smoke notebook checks it.
- **Preview model:** Groq may withdraw it at short notice (`qwen3.6-27b` was replaced within months). The generator code only depends on `base_url` + model name, so a fallback is a config change: another Groq model, or Qwen on Colab served by vLLM.
- **Rate limits** (as shown in the console): 30 requests/min, 1K requests/day, **8K tokens/min, 200K tokens/day**.
  - At ~1.5–2K tokens per example that is **~100–130 examples/day** and ~4 per minute.
  - Plan: run every day from 9 to 20 Nov → **~1,200–1,500 examples**. That is about 10% of the train set and enough for the R4 ablation.
  - Groq's paid Developer tier lists far higher limits for this model (250K tokens/min). At $0.80/M input and $4.00/M output, 4,000 examples ≈ 8M in + 1.6M out ≈ **$13**. Upgrade only if 1,500 proves too few.
- **Saving tokens:** cap contexts at ~350 words (cut at a paragraph boundary), ask for compact JSON without explanations, and use one call per example (both the faithful and the hallucinated version).
- **Native test answers** use the same daily budget (~200 × ~1.2K tokens ≈ 1.5 days). Schedule them on separate days from augmentation.

The generation spec below applies to whichever generator is used.
- **Source:** `alexandrainst/multi-wiki-qa`, config `tr` (5,006 rows, CC BY-NC-SA 4.0). Split **by Wikipedia title**: 4,000 rows for augmentation, ~1,000 reserved as the context pool for the native test set. The two never share a title.
- **Generation (per row, one API call, JSON output):**
  1. a faithful answer in natural Turkish (2–5 sentences) to the question using only the context;
  2. for 50% of rows, a hallucinated version of that answer, with each inserted error wrapped in `<HAL>…</HAL>`. Error type is sampled from: entity swap, number or date change, contradiction of a stated fact, or an added unsupported claim. 1–3 errors per answer.
  3. For one third of rows, a **summary** of the context instead of a QA answer (to strengthen the weak Summary task), with the same faithful/hallucinated scheme.
- **Parsing and validation:**
  - Strip the tags and convert them to character spans.
  - Reject the row if: tags are unbalanced or nested; the text outside the tags differs from the faithful answer by >30% (character-level ratio); the text is not Turkish; there are too few sentences or it is truncated; a "hallucinated" span actually appears verbatim in the context.
- **Quality gate:** generate 30, the user reviews them by eye, and we continue only if ≥85% are acceptable. Otherwise revise the prompt and repeat.
- **Cost:** about 4,000 calls × ~2k input + ~0.4k output tokens ≈ 8M input / 1.6M output tokens. Expect a few US dollars; check current DeepSeek pricing first.
- **Week-1 check:** read DeepSeek's API terms on using outputs for training a model, and record the result in `verification`.
- **Engineering:** async calls with a concurrency limit, retry with backoff, results appended to a JSONL on Drive (so a crash resumes where it stopped), `temperature` 0.7, fixed seed list, model name and date logged per row.
- **As implemented (`hdtr.augment`, notebook 04, 9 Nov):**
  - The detector input copies the **RAGTruth-TR templates**: QA = question + three passages (the ~2,500-character context window around the answer, split into 3 contiguous parts) with the same instruction wording and endings; Summary = "Aşağıdaki haberi N kelimeyle özetleyin:" with N = 50–130. Wording variants are sampled at RAGTruth-TR's rates, so the template cannot reveal which rows are augmented. The generator answers exactly this prompt.
  - Reference/link/category sections are cut. Rows whose answer sits in a list or category line, or whose window is under 400 characters, are skipped (173 of 4,003), leaving **3,830 rows → ~5,700 records** (every faithful answer plus ~1,900 versions with errors).
  - Error types are sampled 40% unsupported claim, 20% each entity, number and contradiction (RAGTruth has more baseless than conflict spans). Labels: unsupported → "Evident Baseless Info", the others → "Evident Conflict".
  - Extra validation: every number in the faithful answer must appear in the prompt (no outside knowledge); refusals are rejected; the tag count must equal the listed errors; a tagged span of ≥4 words that appears verbatim in the context is rejected (short entity or number swaps may reuse context words).
  - A row gets up to 2 attempts. A $4 budget cap (counted at peak price) stops the full run early.
  - The multi-wiki-qa `id` is the article URL and repeats, so plans are keyed by row index.

### 2.7 Native Turkish test set (D7)
- **Contexts (~200):** ~150 from the held-out multi-wiki-qa `tr` pool (Wikipedia), ~50 from `mtntasci/turkish-legal-rag` (CC BY 4.0, a second domain).
- **Questions:**
  - 50% use the dataset's question.
  - 50% are rewritten to ask for details the context only partly covers. This induces hallucination without making the question unanswerable.
- **Answers:** 2 generators, half each:
  - an open model on Colab via vLLM (Qwen2.5-7B-Instruct or Qwen3-8B, Apache-2.0);
  - DeepSeek through the API.
  - Generic RAG prompt. Answers are checked for completeness (length and final punctuation).
  - Target: ~40–50% hallucinated (measured after labeling; top up if far off).
- **Labeling:**
  1. Write `docs/labeling-guide.md` first (Turkish and English, with examples). Key rule: a claim that is true in the world but absent from the context counts as hallucinated.
  2. The user labels example-level yes/no in a CSV or Google Sheet, without seeing any model output. About 1 minute per example.
  3. Claude labels the same CSV independently in a session.
  4. Compute Cohen's κ (human vs LLM), and κ per domain.
  5. The user adjudicates every disagreement, and the user's label is final.
  6. For hallucinated answers, Claude proposes character spans; the user accepts or edits them.
- **Time for the user:** about 3.5 h (labeling) + ~1 h (adjudication) + ~1.5 h (span edits), spread over weeks 2–3.
- **Release:** this set inherits NC-SA from Wikipedia and multi-wiki-qa; the legal part is CC BY. This is documented in the dataset card.

### 2.8 Classical baselines (rubric: "2 classical ML")
1. **Lexical overlap (unsupervised):** for each answer sentence, the max TF-IDF cosine with any context sentence. The sentence is flagged if it falls below a threshold tuned on dev. Spans are whole sentences.
2. **Word-level logistic regression (supervised):** features per answer word:
   - Is the word, or its first 5 characters (a crude Turkish stem), in the context?
   - Is it a number or date, and is it in the context?
   - Capitalized? Position in the sentence.
   - Sentence-level TF-IDF similarity.
   - The same features for neighboring words.

   Trained on the cleaned train set and tuned on dev. Cheap, interpretable, and a meaningful floor.

---

## 3. Phases

Week boundaries run Saturday–Friday. Each phase has an exit criterion and a **cut line**: what we drop if the phase runs late.

### Phase 0: Setup (Fri 6 – Sun 8 Nov)
| Task | Owner | Output |
|---|---|---|
| Create private GitHub repo, push this layout and `docs/` | Claude writes it, user creates the repo | repo |
| `pyproject.toml`, `hdtr` skeleton, `labels.py` + `metrics.py` with unit tests | Claude | passing `pytest` locally |
| Colab Secrets: `GH_TOKEN` (and `DEEPSEEK_API_KEY` once you have it); Drive folder `MyDrive/hdtr/` | User | — |
| `00_setup_smoke.ipynb`: clone, install, load TabiBERT and one Newmind model, run 5 test examples | Claude writes, user runs | `results/00_smoke.json` (GPU, versions, timings) |

**Exit:** the smoke notebook runs end to end on L4.

### Phase 1: Feasibility and foundations (Sat 7 – Fri 13 Nov) → go/no-go on Friday 13 Nov
| Notebook / task | Content | Output |
|---|---|---|
| `01_data_audit` | Load RAGTruth-TR; schema, split × task cross-tab, label ratio; **which text the offsets index**; token lengths under TabiBERT / mmBERT / Newmind tokenizers; alignment to English; audit counts; build `source_id`, the dev/test split and the train/validation split | `results/01_audit.json`, `data/splits/*.json` (id lists) |
| `02_baselines` | Inference of the 6 Newmind models + KR Labs v2 mmBERT on **all 2,700 test responses** (predictions saved; metrics only on dev for now) + classical baselines; timing per model | predictions on Drive, `results/02_baselines_dev.json` |
| `03_pilot` | 1 epoch on the QA subset of cleaned train for TabiBERT, MoganBERT-TR, mmBERT-small, mmBERT-base, KR Labs v2 mmBERT; lr {1e-5, 3e-5} for TabiBERT; weighted CE vs focal on TabiBERT; validation word-F1, minutes, compute units | `results/03_pilot.json` |
| `docs/analysis-plan.md` | Final D1/D2 text: metrics, test split, significance rule, which runs exist | committed **before Phase 3** |
| `docs/labeling-guide.md` + `05a_native_contexts` | Select ~200 contexts, write questions | `data/native/contexts.csv` |
| DeepSeek check | Terms on using outputs for training; pricing; 30-example quality gate (`04_augmentation` in gate mode) | gate result |

**Go/no-go criteria (Friday 13 Nov):**
- [ ] Baselines run and produce numbers within a plausible range of their published figures, or the difference is explained (level or metric definition).
- [ ] Alignment ≥95%, or fallback grouping validated.
- [ ] Extrapolated compute for Phase 3 ≤ 45 compute units.
- [ ] Augmentation quality gate passed (or a revised prompt is in progress).

**Cut line:** if alignment fails and the fallback is weak, drop the tag-count rule (rule 3) and group by fuzzy clusters only. MoganBERT-TR and mmBERT-small are pilot-only; promote MoganBERT-TR in place of TabiBERT only if it wins on validation by >1 F1.

### Phase 2: Data (Sat 14 – Fri 20 Nov)
| Notebook / task | Content | Output |
|---|---|---|
| `01_data_audit` (final) | Freeze the cleaning rules; write the cleaned train and validation sets to Drive; clean-test id list | `results/01_audit_final.json` |
| `04_augmentation` | Full DeepSeek generation (~4,000), validation, rejection stats, examples; Turkish augmentation set on Drive | `results/04_augmentation.json` |
| `05b_native_answers` | Generate answers with Qwen (vLLM) + DeepSeek; completeness checks; labeling CSV | `data/native/to_label.csv` |
| Native labeling round 1 | User labels ~100 examples blind | partial CSV |
| `06_train` (code) | Config-driven training notebook; test-run R1 for 200 steps | `results/06_r1_smoke.json` |

**Exit:** cleaned data frozen, augmentation set ready, `06_train` works, half the native set labeled.
**Cut line:** if augmentation slips, start Phase 3 anyway (R0–R3 don't need it). R4 moves into Phase 3's second half.

### Phase 3: Training and post-processing (Sat 21 – Fri 27 Nov)
| Task | Content | Output |
|---|---|---|
| R1, R2, R3 | Full runs with our recipe on cleaned data (one per Colab session; resume on disconnect) | checkpoints on Drive; `results/06_r{1,2,3}.json` incl. loss and metric curves |
| R0 | Old-style recipe (capped steps if needed) | `results/06_r0.json` |
| Selection | Best of R1–R3 by **dev** example F1 + span F1 (tie → smaller model) | decision noted in `docs/analysis-plan.md` |
| R4 | Best run + augmentation (mixed into train, ≤ 25% of examples) | `results/06_r4.json` |
| Seeds | 3 seeds for R1–R4; report mean ± sd and select on the mean | seed-variance numbers |
| `07_postprocess` | Word snapping, merging, threshold tuning on dev for **all** systems (baselines included) | `results/07_thresholds.json` |
| Native labeling | Round 2 (~100); Claude labels all; κ; adjudication; span edits | `data/native/native_test_v1.jsonl` |

**Exit:** final model chosen; thresholds frozen; native set final.
**Cut line, in drop order:** third seed → R0 at full length (keep the capped version) → R2 → R4 (augmentation becomes "future work"). **R1, R3 and the baselines are never cut.**

### Phase 4: Final evaluation, visuals, report (Sat 28 Nov – Fri 4 Dec)
| Task | Content | Output |
|---|---|---|
| `08_final_eval` | **One run** on final test (full + clean) and the native set: all systems × all levels × all tasks; bootstrap CIs; permutation test; ablation table | `results/08_final.json`, tables as CSV |
| `09_figures` | All figures (list below) from saved predictions | `results/figures/*.png` |
| `main.ipynb` | Deliverable notebook (Section 4) | runs top to bottom in quick mode on T4 |
| Model hosting for submission | Trained model zipped (fp16, ~300 MB) at a shared Drive link via `gdown`, **or** create the HF account now with a private repo turned public on submission day (decide by 30 Nov) | download works in a fresh Colab |
| Report draft | All sections (Section 5) | `docs/report/` draft, PDF ≥10 pages |

**Figures:**
1. Pipeline diagram.
2. Dataset statistics (task mix, label ratio, length histogram per tokenizer, audit funnel).
3. Confusion matrices (example level) for the best baseline and our model.
4. Training loss and validation F1 curves for R0–R4.
5. Model comparison bars per task, with CIs.
6. Precision–recall and ROC curves (example level).
7. Ablation chart.
8. Translated vs native gap chart.
9. t-SNE of answer-token embeddings (5k sampled, colored by gold label) for the base model vs fine-tuned R1.
10. Qualitative examples ("Ocak'a" before and after snapping).

**Exit:** complete report draft; `main.ipynb` runs in quick mode.

### Phase 5: Polish and submit (Sat 5 – Sun 7 Dec, 23:00)
- Proofread the report; check IEEE references; check that every figure is referenced and explained.
- Run `main.ipynb` in a **fresh** Colab session in quick mode and save the outputs into the `.ipynb`.
- Dataset package: split id lists, native test set (+ card: source, license, preprocessing), and a link to or copy of the augmentation set.
- Make the GitHub repo public. Submit the PDF, `main.ipynb`, the dataset (or link) and the slides draft. **Submit on Saturday 6 Dec**; Sunday is the buffer.

### After submission (8 – 25 Dec)
- Slides (20 min), with Q&A prep: faithfulness vs truth, why not GPT-4, why translated data hurts, is the gain significant, why TabiBERT.
- Hugging Face release: model + card + (optional) Space demo; dataset card.

---

## 4. Deliverable notebook (`main.ipynb`)
- **Header:** title, name, student number, course, links.
- **`MODE = "quick"`** (default) or `"full"`.
- **Quick mode** (~10–20 min on T4):
  - downloads saved predictions and the trained model;
  - recomputes every table and figure from predictions;
  - runs live inference on 20 examples, showing the "Ocak'a" demo.
- **Full mode:** runs data prep, the baselines, one training run (R1 config) and evaluation. Marked as needing an L4/A100 and several hours.
- Every step has a markdown explanation and comments. Outputs are saved under each cell. Data loads from public sources (Hub, or a Drive link), never from private paths.

## 5. Report outline (≥10 pages, IEEE)
| Section | Content | ~Pages |
|---|---|---|
| Abstract | 150–200 words | 0.3 |
| Introduction | Problem, RAG faithfulness, why a small encoder, contributions | 1 |
| Related work | LettuceDetect, RAGTruth, Turk-LettuceDetect (incl. reporting issues), generative detectors (RL4HS, SOPO), Turkish encoders (TabiBERT, mmBERT), multilingual hallucination data (PsiloQA, HalluVerse) | 1.5 |
| Data | RAGTruth-TR, alignment and audit funnel, splits, augmentation pipeline, native test set and κ, licenses | 2 |
| Methods | Task formulation, models, recipe, post-processing, baselines, metrics, significance | 2 |
| Results | Main table, per-task table, ablation, native gap, figures | 2.5 |
| Discussion | What worked and why, errors (qualitative), limitations (translated train set, single annotator, LLM-assisted labels, NC licenses) | 1 |
| Conclusion and future work | | 0.5 |
| References | IEEE (verified list in `verification-2026-11-06.md`) | 0.5 |

## 6. Compute budget (Colab Pro, 100 compute units; L4 ≈ 1.7 CU/h)
| Item | Estimate (CU) |
|---|---|
| Smoke + audit | 2 |
| Baseline inference (8 models × 2,700 + native) | 4 |
| Pilot (5 models + 3 recipe variants, 1 QA epoch) | 6 |
| Native answers (Qwen via vLLM) | 1 |
| R1 + R0 (TabiBERT, ~2–3 h each) | 8 |
| R2 + R3 (mmBERT, ~4–5 h each) | 16 |
| R4 + second seeds | 12 |
| Post-processing, final eval, t-SNE, quick-mode check | 5 |
| **Total** | **~54** (≈45% buffer) |

**Measured in the 03 pilot (L4, bf16, 9 Nov):** TabiBERT ~18.5k tokens/s, 3.4 GB peak; mmBERT-base ~15.2k tokens/s, 5.9 GB; mmBERT-small ~29.7k tokens/s.
One full-train epoch is ~12 min for TabiBERT and ~17 min for mmBERT-base, so a 4-epoch run with evaluation is ~1 h (≈ 1.7–2.2 CU).
Revised Phase 3 total with 3 seeds for R1–R4 plus R0: **~25 CU**, well under the 60 CU in the table above. A100 is not needed.

The table above is the original estimate. Rules: pick the runtime type before starting, disconnect as soon as a run ends (idle time burns units), and watch the Resources tab.

## 7. Risks
| Risk | Signal | Response |
|---|---|---|
| Colab disconnect mid-run (no background execution on Pro) | run dies | checkpoint every 500 steps; resume; keep the tab open; one run per session |
| Long inputs cause out-of-memory errors on L4 | OOM in pilot | gradient checkpointing, batch 1 + accumulation, length-grouped batches; A100 only for mmBERT runs |
| Alignment to English fails | <95% pass | fuzzy grouping; drop rule 3 |
| Augmentation quality low | gate <85% | revise prompt; smaller pilot batches; if still bad, R4 becomes future work |
| DeepSeek terms forbid training use | week-1 check | switch generator to Qwen on Colab (same code, OpenAI-compatible vLLM server) |
| Our model doesn't beat the best baseline | dev numbers | honest result; the main contributions (evaluation + native set + controlled comparisons) still stand |
| Native set too small for significance | wide CIs | report CIs, don't overclaim; example-level is primary |
| Schedule slip | phase exit missed | apply that phase's cut line the same day |

## 8. Progress log

**2026-11-08 (Phase 0 and part of Phase 1)**
- Repo scaffolded: `hdtr` package (`text`, `labels`, `postprocess`, `metrics`, `data`, `infer`), 28 unit tests passing, notebooks `00_setup_smoke` and `01_data_audit`.
- Ran the alignment and audit locally on the real data (`results/01_audit_local.json`):
  - **Alignment works by plain row order: 99.99% of rows pass.** The Turkish rows follow the English response file exactly. This recovers `source_id` for all 2,965 sources × 6 responses.
  - Digits agree in only 89.6% of rows, because translation rewrote numbers: "5 pm" → "17:00", added °C conversions, and dropped "in 137 words". So digits are an audit signal, not an alignment criterion.
  - Audit: span-count mismatch 1.0%, truncated 0.03%, **number added by translation 3.5%** (an unlabeled unsupported fact, i.e. label noise; soft flag). Hard drops total **1.06%** (170 train / 19 test rows). Translation noise is lower than the original roadmap feared, so the "+cleaning" ablation may show a small effect. Report that honestly.
  - The first repetition rule flagged business names repeated in Data2txt answers. It was changed to compare against the English original (now 0 drops).
  - Hallucinated share: Data2txt 64–69%, QA 18–31%, Summary 23–31% (test/train).
- `data/splits.json` committed: train 14,334 / validation 756 / dev 540 / test 2,160 (+ `_clean` variants).
- Still to run in Colab: `00_setup_smoke` (GPU, detectors, TabiBERT memory, DeepSeek call) and the tokenizer-length part of `01`.

**2026-11-09**
- Generator switched from Groq (200K tokens/day limit, which would mean running daily for ~12 days) to DeepSeek `deepseek-flash`. It runs as one batch for an estimated $2.50–5. Notebook 00 now checks the DeepSeek key, JSON mode, and that thinking mode is off.
- **Notebook 00 ran on Colab L4** (`results/00_smoke.json`):
  - Environment: Python 3.13, torch 2.14, transformers 4.57.6, bf16 available, no flash-attn (ModernBERT falls back to SDPA without unpadding, so length-grouped batching matters).
  - **Label offsets index the `answer`:** 99.8% of spans fall on word boundaries in the answer vs 15% in the prompt.
  - TabiBERT and mmBERT keep case. The Newmind ModernBERT tokenizer lowercases internally, but its offsets are still valid.
  - Both detectors run at ~0.04 s/example, so the 2,700 test responses take ~2 min per model. Newmind's label 1 = hallucinated (its one prediction matched gold exactly). KR Labs v2 mmBERT predicted nothing on 5 examples (zero-shot, expected to be weak).
  - **TabiBERT training memory is small:** 8,192 tokens fwd+bwd at batch 1 with gradient checkpointing = 4.8 GB. 2,048 tokens = 1.4 GB, 0.1 s.
  - DeepSeek: valid JSON, no reasoning content, 22 output tokens. Thinking-off works.
- **Token lengths, measured locally** (`results/01_token_lengths_local.json`, prompt + answer):

  | Tokenizer | mean | p99 | max | > 512 | > 4,096 |
  |---|---|---|---|---|---|
  | TabiBERT | 868 | 2,028 | 3,850 | 78% | 0% |
  | mmBERT | 997 | 2,516 | 4,591 | 92% | 0.01% |
  | Newmind ModernBERT-tr | 813 | 1,932 | 3,390 | 75% | 0% |
  | BERTurk | 846 | 2,010 | 3,511 | 77% | 0% |

  - A 512-token model would truncate ~3/4 of inputs, which shows why long context matters. `max_length=4096` covers every TabiBERT input.
  - mmBERT needs ~15% more tokens than TabiBERT on this data (the TabiBERT paper's 41% is for general text).
  - One TabiBERT epoch over train is ~13.1M tokens. Extrapolating from the smoke test, that is roughly 10–20 min/epoch on an L4, much cheaper than the §6 estimate. The pilot will measure it.
- **Finding: 9% of TabiBERT tokens span two words** (e.g. `a kadar` = the suffix of "Ocak'a" + "kadar"). For mmBERT and Newmind it is 0%. A cross-word token carries one label, so predictions can bleed into the next word and gold boundaries get coarser.
  - Pilot (03) will compare default tokenization with word-split input (`is_split_into_words`) for TabiBERT.
  - Good material for the report: a Turkish-tokenizer effect on span detection.
- **Classical baselines on dev** (local CPU, τ=0.5, `results/02_classical_dev_local.json`):
  - Lexical overlap: example F1 0.57, span F1 0.10, example AUROC 0.75.
  - Word logistic regression: example F1 0.60, span F1 0.15, example AUROC 0.74, word AUROC 0.73.
  - Strongest features: share of words in the sentence not found in the context, and "is a number". Data2txt example F1 0.78 equals predicting "hallucinated" for everything (64% of Data2txt dev is hallucinated), so example F1 must be read next to AUROC.
- Notebook `02_baselines` written: all 7 transformer detectors + 2 classical baselines on dev + sealed test, with predictions saved to Drive.
- **Notebook 02 ran on L4** (`results/02_baselines_dev.json`). All 9 systems worked. Dev, τ = 0.5, n = 540:

  | System | ex-F1 | span-F1 | ex-AUROC | word-AUROC | Summary span-F1 |
  |---|---|---|---|---|---|
  | Newmind TurkEmbed4STS-HD | **0.700** | **0.478** | **0.834** | **0.880** | 0.234 |
  | Newmind bge-m3-HD | 0.681 | 0.440 | 0.828 | 0.881 | 0.126 |
  | Newmind EuroBERT-tr | 0.668 | 0.420 | 0.820 | 0.859 | 0.113 |
  | Newmind ModernBERT-tr (paper's best) | 0.668 | 0.467 | 0.801 | 0.872 | 0.176 |
  | Newmind Ettin-150M-TR | 0.610 | 0.375 | 0.794 | 0.857 | 0.110 |
  | Newmind Ettin-32M-TR | 0.616 | 0.362 | 0.755 | 0.850 | 0.147 |
  | KR Labs v2 mmBERT (zero-shot) | 0.190 | 0.028 | 0.667 | 0.803 | 0.021 |
  | Word logistic regression | 0.603 | 0.155 | 0.738 | 0.734 | 0.122 |
  | Lexical overlap | 0.569 | 0.104 | 0.751 | 0.695 | 0.056 |

  - **Reproduction check passes.** ModernBERT-tr word-level macro F1 is 0.722 on our dev, against the paper's token-level macro F1 0.7266. Same level of performance, so the go criterion for baselines is met.
  - **The paper's "best" model is not the best under one protocol.** TurkEmbed4STS-HD leads on every overall metric. Dev has only 540 responses, so confirm on the final test with CIs before claiming it.
  - **Summary is the weak task for every system** (span F1 0.11–0.23, example AUROC 0.59–0.70). The roadmap's focus on summary-style augmentation is justified.
  - **KR Labs v2 zero-shot** almost never crosses 0.5 (span F1 0.03), yet word AUROC is 0.80. It ranks Turkish tokens sensibly but is badly calibrated, so it is a promising warm start (R3).
  - Example F1 at 0.5 hides calibration differences. Thresholds will be tuned per system on dev (notebook 07).
  - Speed: 0.018–0.047 s/example on L4.
- **Go/no-go (Phase 1):** baselines ✓, alignment ✓, compute ✓ (training is cheap). Still open: the DeepSeek 30-example quality gate. **Decision: go.**
- **Augmentation gate, prompt v1** (`results/04_gate_v1.json`, DeepSeek `deepseek-flash`, $0.021 for 30 rows = $0.0007/row at peak, so ~$2.70 peak / ~$1.35 off-peak for the full 3,830 rows):
  - Automatic checks: 30/30 accepted on the first try.
  - Claude reviewed all 30 against their contexts: **28/30 acceptable (93%, gate threshold 85%)**.
    - Faithful answers: 29/30 correct. #30 says "two years later" for albums from 2000 and 2003 (a computed interval). #7 credits Sean Ellis as producer from an ambiguous Turkish source sentence (borderline).
    - Versions with errors: 13/15 good. #5's added claim is half entailed by the context ("vegetarian" → "eats no meat or fish") and contradicts the next clause. #30 tags "Ankara'daki Kadri Yörükoğlu Lisesi", although only "Ankara'daki" is wrong. Error-type labels are sometimes off (a changed number reported as "unsupported"); binary labels are unaffected.
  - **Position shortcut found:** 10/15 versions with errors end on a tagged span, and 33% of spans are a whole appended final sentence (RAGTruth-TR QA+Summary: 11%). Every faithful answer is also a training example, so a detector could learn "extra last sentence = hallucinated". Unsupported claims were also often vague praise ("büyük ilgi gördü").
  - **Prompt v2:** unsupported claims go mostly inside existing sentences as specific details; errors spread over the answer; tags cover only the wrong words; no computed intervals or counts in faithful answers. `batch_stats` now reports `span_final_sentence_share`. Output files are versioned, so gate v2 starts fresh. Gate v2 is to be rerun, with the user spot-checking 5 rows.
- **Augmentation gate, prompt v2** (`results/04_gate_v2.json`, same 30 contexts, $0.023 = $0.00077/row, so ~$2.95 peak / ~$1.50 off-peak for the full run):
  - Automatic checks: 30/30 accepted (2 on the second attempt). **`span_final_sentence_share` 0.111** (v1 0.333; RAGTruth-TR 0.11): the position shortcut is gone.
  - Claude's review: **28/30 acceptable (93%)**. Errors are now mostly subtle details inside sentences ("kışlık olarak saklanan kurutulmuş et", "üyelerinin yoğun isteği üzerine"), and all v1 defects are fixed.
    - #29 `mwqa00669`: a tag around an *unchanged* name ("Kosta Runjaić'e"), which would label correct text as hallucinated. New check `hallucinated_span_unchanged`: every span must overlap text that differs from the faithful answer (character diff). Replayed on both gates, it flags exactly this row.
    - #13 `mwqa01400`: the inserted clause is ungrammatical ("Hollanda'nın organizasyonu … gerçekleştiği açıklandı"). Not checkable automatically. Watch for it in the full-run sample.
    - Borderline #12 `mwqa02799`: the faithful summary reads the results table's total row as "2 MPs elected with 100% of the vote". Garbled, but not against the source.
  - **Gate passed.** The user spot-checks 5 rows (#11, #12, #13, #14, #29), then runs the full batch.
- Notebook `04_augmentation` and `hdtr.augment` written (spec in §2.6), with 11 unit tests (49 in total). A local dry run of both modes with a fake DeepSeek client passes: resume, retries, budget accounting, review cards and CSV, parquet output. Title split committed as `data/mwqa_split.json` (augmentation 4,003 rows / 540 articles; native pool 1,003 rows / 133 articles).
- Notebook `03_pilot` and `hdtr.train` written (HF Trainer, weighted CE / focal loss, word-split encoding option for TabiBERT).
- **Notebook 03 (pilot) ran on L4** (`results/03_pilot.json`, commit 01a50b5). Each run: 1 epoch on train QA (4,737), scored on validation QA (250 answers, 82 hallucinated), τ = 0.5.

  | Run | ex-F1 | span-F1 | ex-AUROC | word-AUROC | span-F1 at best τ | min / QA epoch | peak GB |
  |---|---|---|---|---|---|---|---|
  | TabiBERT lr 3e-5 | **0.734** | **0.667** | **0.883** | 0.930 | 0.668 (0.4) | 2.2 | 3.4 |
  | mmBERT-small | 0.702 | 0.664 | 0.869 | 0.918 | 0.664 (0.5) | 1.9 | 3.0 |
  | KR Labs v2 mmBERT* | 0.716 | 0.661 | 0.877 | **0.937** | 0.667 (0.4) | 3.6 | 5.9 |
  | mmBERT-base | 0.704 | 0.658 | 0.868 | 0.925 | 0.662 (0.4) | 3.6 | 5.9 |
  | TabiBERT lr 1e-5 | 0.711 | 0.640 | 0.854 | 0.917 | 0.640 (0.5) | 2.2 | 3.4 |
  | TabiBERT lr 2e-5 | 0.690 | 0.638 | 0.866 | 0.928 | 0.638 (0.5) | 2.6 | 3.4 |
  | MoganBERT-TR | 0.683 | 0.637 | 0.872 | 0.923 | 0.639 (0.4) | 2.2 | 3.3 |
  | TabiBERT word-split | 0.707 | 0.626 | 0.874 | 0.916 | 0.626 (0.4) | 2.7 | 3.5 |
  | TabiBERT focal | 0.574 | 0.480 | 0.847 | 0.912 | 0.618 (0.4) | 2.2 | 3.4 |

  \* KR Labs v2 saw the English versions of our validation answers (validation is carved from RAGTruth train), so its row is optimistic. It is compared on dev only.
  - **Every base model is in the same band (span F1 0.63–0.67).** With one seed and 82 hallucinated answers, gaps below ~0.03 span F1 are not reliable. So all three planned bases go to full runs, with 3 seeds each.
  - **Learning rate:** 3e-5 > 2e-5 ≈ 1e-5 on every metric. One epoch favours a higher rate, but early stopping protects longer runs, so 3e-5 is used for all runs.
  - **Focal loss is dropped.** It shifts probabilities down (span F1 0.48 at τ 0.5, 0.62 at τ 0.4) and has lower AUROC than weighted CE.
  - **Word-split input is dropped.** It gave no gain (0.626 vs 0.638) and costs 19% more tokens. For the report: the cross-word tokens (9% of TabiBERT tokens) do not measurably hurt span detection after fine-tuning.
  - **MoganBERT-TR stays pilot-only.** It ties TabiBERT, short of the >1 F1 needed to replace it.
  - **mmBERT-small equals mmBERT-base on QA** at half the cost. It is an optional cheap extra in Phase 3; the "tie → smaller model" rule applies.
  - **KR Labs v2 starts far ahead:** first logged loss 0.78 vs 2.25 for plain mmBERT-base with the same architecture, and it has the lowest final loss and the best word AUROC. Its hallucination pretraining transfers, but part of the head start may be its English exposure to our train set.
  - **Training is much cheaper than planned** (see §6): ~12 min per full epoch for TabiBERT.
  - The pilot covers QA only. Summary, the weak task in the baselines, is first measured in the full runs.

## 9. Immediate next steps (from 9 Nov)
1. **User:** spot-check gate v2 rows `mwqa00879`, `mwqa02799`, `mwqa01400`, `mwqa02379`, `mwqa00669` (agree or disagree with Claude's verdicts).
2. **User:** run `04_augmentation` with `MODE = "full"`, `GATE_PASSED = True`, off-peak (~$1.50; $4 cap), and send `results/04_augmentation.json`.
4. **Claude:** write `06_train` (config-driven R0–R4, 3 seeds, checkpoints on Drive, resume) and `docs/analysis-plan.md`. Full runs can start early because training is cheap.
5. Native test set: context selection from the reserved titles, labeling guide, answer generation (`05b`).
