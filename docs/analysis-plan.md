# Analysis Plan

- **Written:** 2026-11-09. No model of ours has been scored on the final test yet. The git commit that adds this file is the timestamp.
- **Rule:** the final test is scored once, in notebook 08, by exactly the procedure below. Any later change goes into §10 (Amendments) with its date and reason, and the original version stays in git history.
- **Scope:** this file fixes decisions D1 and D2 of `technical-roadmap.md` and adds the selection and testing rules for the training runs.

---

## 1. Questions

| # | Question | Comparison | Status |
|---|---|---|---|
| Q1 | Does our detector beat the best existing Turkish detector? | final model vs best baseline, final test | **primary (confirmatory)** |
| Q2 | Turkish-native vs multilingual base model | R1 (TabiBERT) vs R2 (mmBERT-base), same recipe | secondary |
| Q3 | Does multilingual *hallucination* training transfer to Turkish? | R3 (LettuceDetect v2 mmBERT) vs R2 (plain mmBERT-base): same architecture and size, only the starting checkpoint differs | secondary |
| Q4 | Effect of our recipe and data cleaning | R1 vs R0 (same base model) | secondary |
| Q5 | Effect of native Turkish augmentation | R4 vs its base run | secondary |
| Q6 | How do detectors do on natural Turkish rather than translated text? | all systems on the native test set | descriptive |

Secondary comparisons are reported with confidence intervals and **without** correction for multiple comparisons. They are labelled as secondary in the report. Only Q1 supports a "better than" claim.

## 2. Data

| Set | Size | Used for |
|---|---|---|
| Train | 14,334 responses (`train_clean`: 14,176) | training. R0 uses `train` (uncleaned); R1–R4 use `train_clean` |
| Validation | 756 / 744 clean, 5% of train sources | early stopping and checkpoint choice within a run only |
| Dev | 540 responses = 90 sources (30 per task), from the official test split | thresholds, choosing between runs, all development decisions |
| **Final test** | **2,160 responses = 360 sources**, official test minus dev | the reported results, scored once |
| Clean final test | 2,146 (audit hard flags removed) | secondary report next to the full final test |
| Native test | ~200 natural Turkish examples (Wikipedia + legal) | Q6, scored once, never used for training or tuning |

- Splits are grouped by **source**: all 6 responses to one source are in the same set. They are fixed in `data/splits.json` (seed 482).
- **Caveat for R3 and the KR Labs baseline:** their starting checkpoint saw the English RAGTruth train set, so their *validation* scores are optimistic. Dev and test come from the official test split, which they did not see.
- Augmentation (R4 only): up to one third of the train size from `mwqa_aug.parquet` (≤25% of the mix). It comes from Wikipedia articles disjoint from the native test set's articles (`data/mwqa_split.json`).

## 3. Metrics

**Primary:** for the *hallucinated* class, on the final test, overall:
1. **Example-level F1:** an answer is predicted hallucinated if it has at least one predicted span after post-processing; the gold label is hallucinated if it has at least one gold span.
2. **Span-level F1:** character overlap, as in LettuceDetect. Precision = overlapping characters / predicted characters; recall = overlapping characters / gold characters.

**Secondary:** word-level F1 and macro F1, example and word AUROC, span precision and recall, and every metric per task (QA, Data2txt, Summary).

- All metrics are computed from per-character probabilities (`technical-roadmap.md` §2.2), so models with different tokenizers are scored on the same words and characters.
- Post-processing (§2.4: word snapping, merging, minimum length 2) is identical for every system.

## 4. Thresholds, checkpoints and run selection

1. **Thresholds** (every system, baselines included):
   - Tuned on **dev** over the grid 0.05, 0.10, …, 0.95; a tie goes to the value closest to 0.5.
   - τ_example maximises example F1 and τ_span maximises span F1.
   - On the final test, example metrics use τ_example and span and word metrics use τ_span. Results at τ = 0.5 are reported as well.
   - Thresholds are frozen in `results/07_thresholds.json` before the final test is scored.
2. **Checkpoint within a run:**
   - R1–R4: the checkpoint with the best validation word F1, evaluated every half epoch, with early stopping after 2 evaluations without improvement and at most 4 epochs.
   - R0: the last checkpoint after 6 epochs.
3. **R4 base:** among R1–R3, the run with the highest dev **selection score**: the mean over its 3 seeds of (example F1 at τ_example + span F1 at τ_span) / 2. If the top two differ by less than 0.01, the smaller model wins (TabiBERT 149M < mmBERT 307M).
4. **Final model** = R4 if R4's dev selection score ≥ its base's; otherwise the base run. This is decided on dev, so augmentation cannot be picked or dropped after seeing the test.
5. **Best baseline** = the existing detector with the highest dev selection score (one run each). Classical baselines are reported but not candidates. On dev, TurkEmbed4STS-HD leads at τ = 0.5; the rule is applied again after threshold tuning.

## 5. Seeds and uncertainty

- **Seeds:** R1–R4 use 3 seeds (42, 1234, 2026); R0 uses seed 42 only. Test results are reported as mean ± sd over seeds, plus each seed's numbers in the appendix.
- **Bootstrap:** 2,000 replicates, resampling **sources**, so all responses to one source move together. For a system with several seeds, each replicate computes the metric for every seed on the same resampled sources and averages over seeds (seed-averaged metric). Reported as a 95% percentile CI.
- **Q1 decision rule** (for each primary metric separately):
  - **Our final model is better than the best baseline** if the 95% bootstrap CI of the difference (ours seed-averaged − baseline) on the final test excludes 0 and is positive.
  - Also reported: a paired permutation test (10,000 permutations, swapping per source) for each seed. The claim is robust if all 3 seeds give p < 0.05.
- **Q2–Q5:** the same seed-averaged bootstrap CI of the difference, reported as an effect size with its CI.
- **Clean final test:** the same numbers, reported next to the full final test. Conclusions are drawn from the full final test.

## 6. Native test set (Q6)

- **Labels:**
  - Example level: the user's label after adjudication, which is the reference.
  - Spans: proposed by Claude, edited by the user.
  - Human–LLM agreement is reported as Cohen's κ, overall and per domain.
- **Primary:** example F1 and example AUROC.
- **Secondary:** span F1, because the spans are LLM-proposed (stated as a limitation).
- **Thresholds:** the dev thresholds of each system (no tuning on native data).
- **Systems:** the best baseline, the other Newmind detectors, and R1–R4. Use all seeds whose weights were kept; if only seed 42 is kept for a run, report it as seed 42.
- **Uncertainty:** bootstrap over contexts (questions about one context move together), 2,000 replicates.
- **Translated vs native gap:** the same system's metric on the final test vs the native test, descriptive with both CIs.

## 7. Failures and exclusions

- A run that crashes (out of memory after the automatic batch-size retries, a NaN loss, a Colab disconnect) is restarted once with the same seed and settings. If it fails again, it is reported as failed.
- **No run, seed or example is dropped because of its scores.**
- Examples are never removed from the final test, except in the separately reported clean test.
- Baselines that fail to load are reported as missing, with the error.

## 8. Order of work

1. Train R0–R3 (notebook 06), with dev scores only.
2. Apply the R4 base rule (§4.3), train R4, and apply the final-model rule (§4.4).
3. Notebook 07: tune and freeze thresholds for all systems on dev (`results/07_thresholds.json`).
4. Notebook 08: score the final test **once** (all systems, full and clean), plus bootstrap and permutation tests per §5.
5. Native test set: label, then score once (§6).

If a bug is found after step 4, the fix and the rerun are documented in §10, with the scores before and after the fix.

## 9. Planned tables and figures

1. **Main table:** all systems on the final test. Primary metrics with 95% CIs, secondary metrics, and our runs as mean ± sd.
2. **Per-task table:** QA, Data2txt, Summary.
3. **Q2–Q5 effect sizes** with CIs (forest plot).
4. **Native vs translated** test, per system.
5. **Threshold curves on dev** (example and span F1 vs τ).
6. **Training curves:** loss, and validation word F1 per seed.
7. **Error analysis:** 20 false negatives and 20 false positives of the final model on dev (not test), categorised by hand.

## 10. Amendments

*(none yet)*
