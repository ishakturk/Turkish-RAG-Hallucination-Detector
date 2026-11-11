# Turkish RAG Hallucination Detector

BIL482 (Natural Language Processing) term project, İstanbul Medeniyet University, fall 2026.
İshak Türk — 23120205003.

A RAG system answers questions from retrieved passages. An answer **hallucinates** when it says something
the passages do not support. This project trains a small encoder model that reads
`(context + question, answer)` and marks the unsupported parts of a **Turkish** answer, and asks how well
such detectors really work.

**Start here:** [`notebooks/main.ipynb`](notebooks/main.ipynb) (open in Colab → Run all; free T4, ~10 min).
It rebuilds every table and figure, recomputes the native-set results from saved predictions, and runs
the released model live.

## Results

| | Example F1 | Span F1 |
|---|---|---|
| **Final model** (mmBERT-base, cleaned RAGTruth-TR, weighted loss), translated final test | **0.766** [0.738, 0.792] | **0.538** [0.493, 0.578] |
| Best existing detector chosen on dev (TurkEmbed4STS) | 0.696 | 0.438 |
| Difference (pre-registered Q1) | **+0.070** [0.049, 0.093] | **+0.100** [0.075, 0.125] |

- **One protocol for all detectors:** the 7 existing Turkish detectors and 2 classical baselines were scored
  on the same characters, with thresholds tuned on a dev split and a final test opened once. The rules
  were written in [`docs/analysis-plan.md`](docs/analysis-plan.md) before testing.
- **The final model beats every existing detector** in exploratory comparisons. Controlled runs show that
  most of the gain comes from the multilingual base model. Hallucination pretraining, the training recipe
  and augmentation matter little on the translated test.
- **New native Turkish test set** (184 answers by Qwen3-8B and DeepSeek). Every trained detector drops
  sharply on it: the final model goes from 0.766 to 0.458 example F1. The model trained with natural
  Turkish augmentation data is best there (0.592). Translated benchmarks overstate how well Turkish
  detectors work.

Final model: [`ishakturk/turkish-rag-hallucination-detector`](https://huggingface.co/ishakturk/turkish-rag-hallucination-detector)
(run `r2_mmbert_s42`, frozen thresholds in its model card).
Report (Turkish, IEEE, 10 pages): [`report/rapor_tr.pdf`](report/rapor_tr.pdf). Article (English, 5 pages):
[`report/article_en.pdf`](report/article_en.pdf). Data card: [`data/README.md`](data/README.md).

## Layout

```
notebooks/main.ipynb   Deliverable notebook (quick mode: results + live model; full mode: retrain and evaluate)
notebooks/00–11        The notebooks that produced the results, each run once in Colab (sources in notebooks/src/)
src/hdtr/              Python package: labels, metrics, post-processing, data, training, evaluation, figures
configs/               Training settings of runs R0–R4
data/                  Splits, the native test set, its labels and spans, saved native predictions
results/               JSON outputs of every notebook (all numbers in the report come from here)
report/                Turkish report, English article (LaTeX + PDF) and figures
docs/                  Roadmap, pre-registered analysis plan, labeling guide, research notes
tests/                 pytest unit tests
```

| Notebook | What it did |
|---|---|
| 00 | Environment and smoke test |
| 01 | Data audit (alignment with the English RAGTruth, rule-based cleaning) and source-grouped splits |
| 02 | Existing detectors and classical baselines on dev |
| 03 | Pilot: base model and recipe |
| 04 | Native Turkish augmentation data generated with DeepSeek |
| 05a, 05b | Native test set: questions, then answers from Qwen3-8B and DeepSeek |
| 06 | Training R0–R4 (3 seeds each, R0 one seed) |
| 07 | Thresholds frozen on dev and the final model chosen by rule |
| 08 | Final test, scored once (Q1–Q5, bootstrap CIs, permutation tests) |
| 09 | Figures and exploratory comparisons |
| 10 | Native test set, scored once |
| 11 | Release: native test file, native predictions, final model on Hugging Face |

## Running

Locally (unit tests, no GPU):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,colab]"
.venv/Scripts/python -m pytest
```

Notebooks are written as percent-format scripts in `notebooks/src/` and converted with
`python scripts/build_notebooks.py`. Notebooks 00–11 need the author's Google Drive and Colab secrets.
`main.ipynb` needs neither.

## License

Code: MIT ([`LICENSE`](LICENSE)). Native test set: CC BY-NC-SA 4.0, for evaluation only (see the data card).
Third-party data keeps its own license (RAGTruth-TR: MIT, multi-wiki-qa: CC BY-NC-SA 4.0,
turkish-legal-rag: CC BY 4.0).
