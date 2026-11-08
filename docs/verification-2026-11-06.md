# Verification of roadmap facts (2026-11-06)

Every fact below was checked against its primary source (arXiv full text, ACL Anthology, Hugging Face cards/API, dataset viewer, GitHub, PyPI) on 2026-11-06.
Status: **OK** = confirmed, **FIX** = the roadmap is wrong or imprecise, **OPEN** = still unknown, check in the week-1 notebook.

## 1. Corrections to the roadmap

| # | Roadmap says | Verified | Status |
|---|---|---|---|
| 1 | RAGTruth: 17,790 train / 2,700 test | 2,965 source instances, 17,790 responses **in total**. Test = 450 instances = 2,700 responses. Train = 15,090 responses (derived; the paper says "the rest"). The Turk paper and all Newmind cards repeat the wrong "17,790 train". | FIX |
| 2 | RAGTruth-TR has separate splits / maybe a validation split | `newmindai/RAGTruth-TR`: one HF split (`train`, 17,790 rows), config `default`. A **column** `split` holds `train` = 15,090 and `test` = 2,700. No validation split. | FIX |
| 3 | Canonical repo might be `newmindai/turk_lettucedetect` (config `tr-ragtruth`) | Returns 401 (private, gated or missing). Config `tr-ragtruth` does not exist. Canonical: `newmindai/RAGTruth-TR`. | FIX |
| 4 | Data audit: count `<HAL>` tag mismatches | Dataset has **no English source and no `<HAL>` tags**. Schema: `prompt`, `answer`, `labels` (list of `{start, end, label}`), `split`, `task_type`, `dataset`, `language`. Any comparison with the English original requires aligning rows to the original RAGTruth first. | FIX |
| 5 | Group validation split by source | There is **no source/document id**. The 6 responses per source were translated separately, so their prompts differ slightly. Grouping must be rebuilt (align to English RAGTruth, or fuzzy-match prompts). | FIX |
| 6 | Newmind code at github.com/newmind-ai/Turk-LettuceDetect | 404. No public evaluation code. | FIX |
| 7 | Paper sizes 135M / 210M / 305M | From the model cards (paper gives no mapping): ModernBERT-tr-HD 135M, EuroBERT-tr 212M, TurkEmbed4STS-HD 305M. | OK |
| 8 | LettuceDetect "about 14.8% better than Luna" | The abstract says 14.8%, but its own Table 2 gives 79.22 vs 65.4 = 13.8 points (21.1% relative). Quote it as "13.8 F1 points". | FIX |
| 9 | Input lengths (mean 801, median 741, 194–2,632) are RAGTruth stats | These come from the **LettuceDetect** paper (Sec. 3), not RAGTruth. Cite LettuceDetect. | FIX |
| 10 | LLM judges: "Mistral Small 3.1/3.2", "QwQ-32B best QA AUROC" | Paper says "Mistral models" (no "Small"); "QwenQ-32B … highest AUROC (0.7267)" in QA. Qwen3-14B F1 0.7429 confirmed; its level is not stated in the text (figure caption: example-level). | FIX |
| 11 | Newer KR Labs release: "LettuceDetect-mmBERT-base" and "LettuceDetect-Qwen-2B" | Real ids: `KRLabsOrg/lettucedect-v2-mmbert-base` and `KRLabsOrg/lettucedect-v2-qwen-2b` (June 2026, Apache-2.0). **No Turkish**: languages en, de, fr, es, it, pl, zh (+ PsiloQA languages for Qwen; Turkish not among them). | FIX |
| 12 | Luna, COLING 2025 Industry | Published title differs from arXiv: "Luna: A Lightweight Evaluation Model to Catch Language Model Hallucinations with High Accuracy and Low Cost", pp. 398–409. | FIX |
| 13 | RAG-HAT, 2024 | EMNLP 2024 Industry Track, pp. 1548–1558. No arXiv version. | FIX |
| 14 | RAGTurk "(get the exact citation)" | arXiv:2602.03652, Köse et al., accepted at SIGTURK @ EACL 2026. | FIX |
| 15 | Colab Pro includes background execution | Not stated for Pro; the FAQ lists background execution for higher tiers (Pro+: up to 24 h continuous). Assume **no** background execution. Prices and CU per month could not be read (JS-only page). | OPEN |

## 2. Confirmed facts

- **LettuceDetect** (Kovács & Recski, arXiv:2502.17125): ModernBERT, RAGTruth, large model example-level F1 79.22; Luna 65.4; GPT-4-turbo 63.4; Llama-2-13B FT 78.7; RAG-HAT 83.9; 6 epochs on one A100 (lr 1e-5, batch 8); MIT code and models.
- **Turk-LettuceDetect** (Taş, El Huseyni, Ezerceli, Bayraktar, Terzioğlu; arXiv:2509.17671, 22 Sep 2025):
  - Translation with gemma-3-27b-it via vLLM, about 12 h on one A100. The `<HAL>` count was only a prompt instruction; no programmatic check and no translation validation are described.
  - Training: 6 epochs, lr 1e-5, batch 4, A100 40GB, about 20 min/epoch.
  - Table II values: all confirmed exactly as in the roadmap. It is titled token-level.
  - **AUROC = Recall in all 12 rows.** Explanation: the reported recall is macro recall (balanced accuracy), and AUROC from hard 0/1 predictions equals balanced accuracy.
  - The text's AUROC 0.8966 (EuroBERT, Data2txt) and 0.8833 (ModernBERT, QA) appear nowhere in the table (which gives 0.7218 and 0.7536). Sec. IV-C names a different model as best in Data2txt. Internally inconsistent.
  - F1 is described as macro over both classes.
  - No validation split, checkpoint selection or early stopping is described.
  - Their reference [1] lists the wrong authors for LettuceDetect, and the paper cites Gemma 2, not Gemma 3. Do not copy either.
- **Newmind model cards disagree with the paper**: ModernBERT-tr-HD card "Overall F1 0.7821" vs paper 0.7266; level unlabeled. The bge-m3-HD card compares its token-level F1 (45.46) with other models' unlabeled 71–78%. This strongly supports the "one protocol, re-run everything" contribution.
- **RAGTruth** (Niu et al., ACL 2024): ~18k responses, 6 LLMs, QA (MS MARCO), Data2Text (Yelp), Summarization (CNN/DM + recent news), 4 hallucination types, no official dev split.
- **Temporal Multi-Signal Fusion** (I. Itkin, arXiv:2608.18115): "median span of 5 tokens and a persistence probability P(Ht|Ht−1)=0.902". Confirmed verbatim.
- **SOPO** (Dong et al., arXiv:2610.32212, Oct 2026): 4B model. Its own Table 1, span-level F1 on RAGTruth: SOPO 63.3, LettuceDetect-Large 58.9, o3 51.8, GPT-5 42.9. Note that LettuceDetect's 79.22 is example-level and its span-level score is 58.9. Good slide example of why levels must not be mixed.
- **RL4HS** (arXiv:2510.02173), **Beyond Document Grounding** (arXiv:2607.00895), **Multi-View Attention** (arXiv:2504.04335): all exist with the claimed titles.
- **HalluVerse-M3** (arXiv:2602.06920): English, Arabic, Hindi, Turkish. **HalluVerse25** (arXiv:2503.07833): English, Arabic, Turkish.
- `lettucedetect` on PyPI: latest 0.2.3 (2026-08-13), MIT, Python ≥ 3.11. `HallucinationDetector(method="transformer", model_path=...).predict(context=[...], question=..., answer=..., output_format="spans")` confirmed. `scripts/evaluate.py` exists. Token- and span-level flags are not yet verified.

## 3. Models and licenses

| Model | License | Params | Context | `trust_remote_code` | Card metrics |
|---|---|---|---|---|---|
| newmindai/ModernBERT-tr-uncased-stsb-HD | MIT | 135M | 8,192 | no | F1 0.7821 (level unstated) |
| newmindai/TurkEmbed4STS-HD | MIT | 305M | 8,192 | **yes** (gte custom code) | F1 0.7666 (unstated) |
| newmindai/lettucedect-210m-eurobert-tr-v1 | MIT | 212M | 8,192 | **yes** (EuroBERT) | F1 0.7777 (unstated) |
| newmindai/bge-m3-HD | Apache-2.0 | 567M | not stated | no | example F1 71.12, AUROC 85.91; token F1 45.46 |
| newmindai/ettin-encoder-150M-TR-HD | Apache-2.0 | 150M | not stated | no | example F1 62.64; token F1 38.40 |
| newmindai/ettin-encoder-32M-TR-HD | Apache-2.0 | 32M | not stated | no | example F1 61.35; token F1 38.33 |
| KRLabsOrg/lettucedect-v2-mmbert-base | Apache-2.0 | 307M | 8,192 | no | not Turkish-trained |
| KRLabsOrg/lettucedect-v2-qwen-2b | Apache-2.0 | 2B | — | — | not Turkish-trained |
| newmindai/RAGTruth-TR (dataset) | MIT (card and API agree) | 17,790 rows | — | — | — |

Notes:
- No Newmind card says whether the test responses were used for model selection.
- The Ettin-TR detectors list unlinked bases (`ettin-encoder-32M-TR`, `ettin-encoder-150M-TR`). The original Ettin encoders (jhu-clsp, arXiv:2507.11412) are English-only, so their Turkish pretraining is undocumented.
- The EuroBERT card's usage code loads the ModernBERT model (copy-paste error). Always use exact repo ids.
- Base models: **mmBERT** (`jhu-clsp/mmBERT-base` 307M / `mmBERT-small` 140M, 8,192 ctx, 1,800+ languages incl. Turkish, MIT, arXiv:2509.06888); **EuroBERT** (210m/610m/2.1B, includes Turkish, 8,192 ctx, Apache-2.0, needs `trust_remote_code`, arXiv:2503.05500; KR Labs pins `transformers>=4.48.3,<5`).

## 4. Dataset statistics (HF dataset viewer)

- `task_type`: Data2txt 6,198 / QA 5,934 / Summary 5,658 (split × task cross-tab not yet computed).
- `labels` spans per answer: mean 0.79, median 0, max 14. The exact share of answers with ≥1 span is OPEN.
- Characters: answer mean 820 (max 3,891); prompt mean 2,559, median 2,238, max 13,444. Token lengths per tokenizer are OPEN. Expect most examples to exceed 512 tokens and all to fit in 8,192.
- Label strings seen: "Evident Conflict", "Evident Baseless Info", "Subtle Baseless Info" (presumably also "Subtle Conflict").

## 5. Consequences for the plan

1. **Use the `split` column**: 15,090 train / 2,700 test. Split test into dev-from-test (~20%) and final test (~80%), grouped by source.
2. **Source grouping must be rebuilt** in week 1. First try aligning row order with the original English RAGTruth / LettuceDetect preprocessed data (which has source ids). Fall back to fuzzy prompt clustering (6 responses per source).
3. **Data audit is redesigned.** Without English or tags, the checks are:
   - span offsets inside the answer, and span text not empty or whitespace;
   - spans cutting through a word;
   - English residue, repetition, and the answer/prompt length ratio;
   - after alignment: span count and length ratio vs the English original.
   Confirm in week 1 which string the offsets index (likely `answer`).
4. **New, stronger idea:** `lettucedect-v2-mmbert-base` is already trained for hallucination detection on a base that covers Turkish, but it has never been trained on Turkish data. Use it twice:
   - (a) a zero-shot cross-lingual baseline;
   - (b) the "other starting point" for fine-tuning, which is more original than starting from raw mmBERT and safer than starting from scratch.
   Leakage check first: confirm its training used only the RAGTruth **train** split, because our Turkish test set is a translation of the RAGTruth test set. (OPEN)
5. **Baseline set:** 6 Newmind detectors + 2 KR Labs v2 models (Qwen-2B is optional because of its compute cost) + classical baselines. Pin `transformers<5` because of the custom code.
6. **Contribution wording:** Newmind's evaluation code is not public and its paper and cards disagree. "To our knowledge, the first public, reproducible evaluation of Turkish hallucination detectors" is defensible.
7. **Colab:** assume no background execution on Pro. Keep the tab open, checkpoint every N steps, and read current CU rates in the Resources tab.

## 6. IEEE references (verified fields)

[1] Á. Kovács and G. Recski, "LettuceDetect: A hallucination detection framework for RAG applications," arXiv:2502.17125, 2025.
[2] S. Taş, M. El Huseyni, Ö. Ezerceli, R. Bayraktar, and F. B. Terzioğlu, "Turk-LettuceDetect: A hallucination detection models for Turkish RAG applications," arXiv:2509.17671, 2025.
[3] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, "RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models," in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers), Bangkok, Thailand, Aug. 2024, pp. 10862–10878, doi: 10.18653/v1/2024.acl-long.585.
[4] B. Warner et al., "Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference," arXiv:2412.13663, 2024.
[5] M. Belyi, R. Friel, S. Shao, and A. Sanyal, "Luna: A lightweight evaluation model to catch language model hallucinations with high accuracy and low cost," in Proc. 31st Int. Conf. Comput. Linguistics: Industry Track (COLING), Abu Dhabi, UAE, Jan. 2025, pp. 398–409.
[6] J. Song, X. Wang, J. Zhu, Y. Wu, X. Cheng, R. Zhong, and C. Niu, "RAG-HAT: A hallucination-aware tuning pipeline for LLM in retrieval-augmented generation," in Proc. EMNLP 2024: Industry Track, Miami, FL, USA, Nov. 2024, pp. 1548–1558, doi: 10.18653/v1/2024.emnlp-industry.113.
[7] H. Su et al., "Learning to reason for hallucination span detection," arXiv:2510.02173, 2025.
[8] J. Dong, Z. Bai, P. Zhu, and P. Zhou, "SOPO: Socratic guided policy optimization for span-level hallucination detection," arXiv:2610.32212, 2026.
[9] Á. Kovács et al., "Beyond document grounding: Span-level hallucination detection over code, tool output, and documents," arXiv:2607.00895, 2026.
[10] Y. Ogasa and Y. Arase, "Hallucinated span detection with multi-view attention features," arXiv:2504.04335, 2025.
[11] I. Itkin, "Temporal multi-signal fusion for token-level hallucination detection," arXiv:2608.18115, 2026.
[12] S. K. Köse et al., "RAGTurk: Best practices for retrieval augmented generation in Turkish," in Proc. SIGTURK Workshop @ EACL 2026, arXiv:2602.03652.
[13] S. Abdaljalil, P. Sharma, E. Serpedin, and H. Kurban, "Halluverse-M3: A multitask multilingual benchmark for hallucination in LLMs," arXiv:2602.06920, 2026.
[14] S. Abdaljalil, H. Kurban, and E. Serpedin, "HalluVerse25: Fine-grained multilingual benchmark dataset for LLM hallucinations," arXiv:2503.07833, 2025.
[15] mmBERT: A modern multilingual encoder with annealed language learning, arXiv:2509.06888, 2025. *(authors to fill)*
[16] EuroBERT: Scaling multilingual encoders for European languages, arXiv:2503.05500, 2025. *(authors to fill)*
[17] Seq vs Seq: An open suite of paired encoders and decoders (Ettin), arXiv:2507.11412, 2025. *(authors to fill)*
