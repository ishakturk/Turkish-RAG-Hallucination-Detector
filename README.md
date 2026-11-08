# Turkish RAG Hallucination Detector

BIL482 (Natural Language Processing) term project, İstanbul Medeniyet University, fall 2026.
İshak Türk — 23120205003.

A small encoder model that reads `(context, question, answer)` and marks the parts of the answer
that are **not supported by the context**. The project compares all existing Turkish hallucination
detectors under one reproducible protocol, trains new detectors on a Turkish-native long-context
encoder (TabiBERT), and measures performance on a hand-labeled native Turkish test set.

- Plan: [`docs/technical-roadmap.md`](docs/technical-roadmap.md)
- Fact checks and references: [`docs/verification-2026-11-06.md`](docs/verification-2026-11-06.md)
- Model and data survey: [`docs/model-and-data-research-2026-11-06.md`](docs/model-and-data-research-2026-11-06.md)

## Layout

```
src/hdtr/      Python package: text, labels, metrics, postprocess, data
notebooks/     Colab notebooks (generated from notebooks/src/*.py)
scripts/       Helpers (notebook builder)
tests/         pytest unit tests
data/          Small committed data (split id lists, native test set)
results/       Small JSON/CSV summaries from Colab runs
docs/          Roadmap, research notes, analysis plan, report drafts
```

## Running

Locally (unit tests only, no GPU):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

In Colab: open a notebook from `notebooks/`. The first cells mount Drive, clone this repo with the
`GH_TOKEN` Colab secret and install the package with `pip install -e ".[colab]"`.

Notebooks are written as percent-format scripts in `notebooks/src/` and converted with:

```bash
.venv/Scripts/python scripts/build_notebooks.py
```

## License

Code: MIT (to be added at release). Data and models: see the dataset and model cards; third-party
assets keep their own licenses (RAGTruth-TR: MIT, multi-wiki-qa: CC BY-NC-SA 4.0).
