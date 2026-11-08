# Base models and data: research (2026-11-06)

Goal: choose starting models and training/test data on merit, without depending on Newmind. Newmind's detectors stay in the project only as **baselines to compare against**.

Task constraints that drive the choice:
- Token classification over context + question + answer.
- RAGTruth-TR prompts average about 2,560 characters (max 13,444), so we need a **long-context (8,192) encoder**.
- Must be trainable on one Colab L4 or A100.

## 1. Base encoder candidates

| Model | Params | Context | Turkish | License | Remote code | Notes |
|---|---|---|---|---|---|---|
| **boun-tabilab/TabiBERT** | 149M | 8,192 | Monolingual Turkish (~13% en/code/math) | Apache-2.0 | no | ModernBERT architecture, trained from scratch on Turkish (arXiv 2512.23065, v4 Oct 2026). Cased BPE tokenizer (normalizer `null`, inferred). TabiBench avg 77.58 vs BERTurk 75.96. Token classification 93.42 vs BERTurk 93.67 (within noise). QA +9.55 F1 over BERTurk. **mmBERT needs ~41% more tokens than TabiBERT for the same Turkish text.** Never used for hallucination detection. |
| moganai/MoganBERT-TR | 149M | 8,192 (config) | Turkish | Apache-2.0 | no | Aug 2026 (arXiv 2608.25768). TrGLUE 78.41, TabiBench 77.73 (single seed). Final training phase at 1,024 context, so quality on 2–3k-token inputs is unverified. Only 98 downloads. |
| jhu-clsp/mmBERT-base | 307M (110M non-embedding) | 8,192 | Yes (1,800+ languages) | MIT | no | arXiv 2509.06888. XNLI-tr 73.8. TabiBench avg 78.98 per the TabiBERT paper (above TabiBERT, at 2× the size). Paper: weaker NER (58.2 vs XLM-R 61.4 WikiANN), attributed to inconsistent prefix-whitespace tokens. |
| jhu-clsp/mmBERT-small | 140M | 8,192 | Yes | MIT | no | Cheap option. Most EuroEval hallucination classifiers use it. |
| **KRLabsOrg/lettucedect-v2-mmbert-base** | 307M | 8,192 (inference example uses 4,096) | **Not trained on Turkish** (en, de, fr, es, it, pl, zh + PsiloQA's 14 languages) | Apache-2.0 | no | mmBERT-base already trained for hallucination spans on code + prose data (arXiv 2607.00895). Card: RAGTruth (English) example-F1 0.743, span-F1 0.528; PsiloQA span-F1 0.714. |
| EuroBERT/EuroBERT-210m | 310M actual (210M nominal) | 8,192 | Yes | Apache-2.0 | **yes** | XNLI-tr 66.4 in the mmBERT paper. The EuroBERT paper says it lags XLM-R on NER. Low priority. |
| Alibaba-NLP/gte-multilingual-mlm-base | 306M | 8,192 | Yes | Apache-2.0 | **yes** | Backbone of Newmind's TurkEmbed. Low priority. |
| BAAI/bge-m3 | ~568M | 8,192 | Yes | MIT | no | Heavy for long inputs on an L4. Already a Newmind baseline. |
| LiquidAI/LFM2.5-Encoder-350M | 354M | 8,192 | Yes (15 languages) | LFM Open License (not OSI) | **yes** | July 2026. License and remote code make it a poor fit for a public release. Its KR Labs hallucination detector has no Turkish. |
| dbmdz/bert-base-turkish-cased (BERTurk) | 111M | **512** | Turkish | MIT | no | The classic Turkish encoder. Useful only to show truncation hurts (sliding window). Optional. |
| SindBERT (SIGTURK 2026) | RoBERTa base/large | NOT FOUND (likely 512) | Turkish | MIT | — | RoBERTa architecture. HF id and numbers not found. Not a candidate. |

**Leakage check, resolved:** KR Labs prose training data has 78,882 training examples = PsiloQA 63,792 + RAGTruth 15,090. That is exactly RAGTruth's train size, and its 2,700 RAGTruth rows sit in the test split. So `lettucedect-v2-mmbert-base` never trained on RAGTruth test. This is inferred from the counts; the card does not state it explicitly.

## 2. Recommended starting points

| Run | Start | Role |
|---|---|---|
| **R1** | TabiBERT | Main model. Turkish-native, efficient tokenizer, 149M (cheap), Apache-2.0, never applied to this task. |
| **R2** | mmBERT-base | Multilingual control at the same architecture: Turkish-native vs multilingual base. |
| **R3** | lettucedect-v2-mmbert-base | Same base as R2 plus multilingual hallucination training: does task knowledge transfer to Turkish? |
| R0 (ablation) | TabiBERT, Newmind-style recipe (fixed 6 epochs, no validation, plain CE, uncleaned data) | Isolates the effect of our recipe and data cleaning. Short pilot-length run if compute is tight. |
| Pilot only | MoganBERT-TR, mmBERT-small | 1-epoch QA pilot. Promote MoganBERT-TR if it beats TabiBERT. |

Questions the design answers:
1. Turkish-native vs multilingual base (R1 vs R2).
2. Value of multilingual hallucination pretraining (R3 vs R2).
3. Value of our recipe and data (R1 vs R0).
4. Everything vs the six Newmind detectors on one protocol.

## 3. Data

### Training data
- **RAGTruth-TR** (MIT): still the only sizable Turkish span-labeled set. 15,090 train responses. Cleaning per the verification notes.
- **No native Turkish span-labeled hallucination dataset exists.** Checked: PsiloQA (14 languages, no tr), Mu-SHROOM (14, no tr), multi-wiki-qa-synthetic-hallucinations (216 configs, **no tr**), EuroEval/alexandrainst classifiers (no tr). HalluVerse-M3 has 1,046 Turkish rows but is not context-grounded for QA and has instance-level labels only.
- **Native Turkish augmentation route:** `alexandrainst/multi-wiki-qa`, config `tr` (5,006 Turkish Wikipedia passages with questions and answers; **CC BY-NC-SA 4.0**).
  - Inject hallucinations with the same pipeline that built the 216-language synthetic set (github.com/alexandrainst/factuality_eval, based on LettuceDetect's RAGFactChecker; factual, temporal and numerical edits; character spans).
  - Needs an LLM to do the rewriting: an open model on Colab or a cheap API.
  - Caveat: NC-SA means derived data must stay non-commercial and share-alike. Fine for coursework; it constrains the license of the released dataset.
- Optional for the mmBERT runs: PsiloQA (CC BY-4.0, 14 languages) as extra multilingual span data.

### Native Turkish test set
- **Contexts:**
  - Turkish Wikipedia (CC BY-SA), or a held-out slice of multi-wiki-qa `tr` (kept disjoint from the augmentation data).
  - `mtntasci/turkish-legal-rag` (CC BY 4.0, 290 verified QA pairs over 25 statutes) as a second domain.
  - Avoid TQuad-2 (license not stated) unless a license is found.
- **Answers:** 2 open instruct LLMs (4–9B), with prompts designed to yield ~40–50% hallucinated answers.
- **Labels:** user labels blind at example level → LLM labels independently → κ → user adjudicates disagreements → LLM proposes spans for hallucinated answers, user edits.
- **Caution:** arXiv 2607.13707 (Turkish/English faithfulness corpus) reports a truncation bug that silently corrupted its LLM-generated answers. Check every generated answer for length and completeness.

## 4. Open checks for the week-1 notebook
- TabiBERT: confirm it is cased, and check how the apostrophe in "Ocak'a" is tokenized.
- Token-length distribution of RAGTruth-TR under the TabiBERT and mmBERT tokenizers.
- MoganBERT-TR quality at 2–3k tokens (pilot).
- factuality_eval: can it run with an open model on Colab, and what does it cost per 1k examples?

## Sources
- TabiBERT: https://huggingface.co/boun-tabilab/TabiBERT, arXiv 2512.23065
- MoganBERT-TR: https://huggingface.co/moganai/MoganBERT-TR, arXiv 2608.25768
- mmBERT: https://huggingface.co/jhu-clsp/mmBERT-base, arXiv 2509.06888
- KR Labs v2: https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base, https://huggingface.co/datasets/KRLabsOrg/lettucedetect-prose-hallucination, arXiv 2607.00895
- EuroBERT: arXiv 2503.05500. LFM2.5-Encoder: https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M
- PsiloQA: https://huggingface.co/datasets/s-nlp/PsiloQA, arXiv 2510.04849
- MultiWikiQHalluA: arXiv 2605.02504; https://huggingface.co/datasets/alexandrainst/multi-wiki-qa-synthetic-hallucinations; https://huggingface.co/datasets/alexandrainst/multi-wiki-qa
- HalluVerse-M3: https://huggingface.co/datasets/sabdalja/HalluVerse-M3
- SindBERT: https://aclanthology.org/2026.sigturk-1.1/
- Turkish RAG data: https://huggingface.co/datasets/mtntasci/turkish-legal-rag, https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval
