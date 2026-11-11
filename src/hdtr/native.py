"""Native Turkish test set (docs/technical-roadmap.md §2.7, docs/analysis-plan.md §6).

About 200 QA examples written in Turkish from the start, so detectors can be scored on natural
rather than translated text (Q6):

- contexts: Wikipedia articles reserved in ``data/mwqa_split.json`` (never used for augmentation)
  and articles of Turkish laws (``mtntasci/turkish-legal-rag``, CC BY 4.0);
- questions: half are the dataset's own question, half are rewritten by DeepSeek to also ask for
  one detail the context does not contain (to invite hallucination without making the question
  unanswerable);
- answers: half from Qwen3-8B on Colab, half from DeepSeek, both given the RAGTruth-TR QA prompt
  the detectors were trained on, so the input format matches training;
- labels: the user, blind (docs/labeling-guide.md). Nothing here is ever used for training or tuning.

Each domain is balanced over the four cells (question type × generator). Candidates are drawn
about 1.5× oversized in a fixed, seeded order; the final set takes, per cell, the first
candidates whose question and answer pass objective checks. No label or score is involved.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path

import pandas as pd

from . import augment as A
from .data import _english_share
from .text import ends_like_sentence, words

SEED = 482
# Bump when REWRITE_SYSTEM changes: log files are named by it, so old rewrites are never reused.
REWRITE_VERSION = 1
LEGAL_DATASET = ("mtntasci/turkish-legal-rag", "qa_benchmark", "test")
# Only rows with human-written questions. The other 77 active rows ("Scenario C Expansion", no
# source_origin) have machine-templated, ungrammatical questions such as "Değişik aynen kabul uyarınca
# öngörülen hukuki süreç nasıl işlemektedir?" (found in the 05a review, 11 Nov, before any answer).
LEGAL_SOURCE = "kaggle_batuhankalem"
REVIEW_FILE = "data/native_review.json"
SIZES = {"wikipedia": 152, "legal": 48}          # final examples per domain, divisible by len(CELLS)
CELLS = (("original", "qwen"), ("original", "deepseek"), ("partial", "qwen"), ("partial", "deepseek"))
OVERSAMPLE = 1.5
# Legal rewrites fail review far more often (14 of 31 in the second 05a run: the "missing" detail is in
# the article, or nothing missing is asked), so legal draws 120 candidates. The first 72 are unchanged.
LEGAL_OVERSAMPLE = 2.5
MAX_CONTEXT_CHARS = 1500   # keeps labeling near 1.5 min per example; legal articles are ≤ 1,400
QWEN_MODEL = "Qwen/Qwen3-8B"
# Qwen3 non-thinking sampling settings recommended by its model card.
QWEN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
DEEPSEEK_TEMPERATURE = 0.7

_REFUSAL_RE = re.compile(r"(cevap|yanıt) verilemiyor", re.IGNORECASE)
# Phrases that point at the source ("metin" alone is also a common first name, so it is not listed).
_META_RE = re.compile(r"\b(pasaj\w*|metinde|metne göre|bağlamda|kaynakta|kaynağa göre|kaynaklara göre)",
                      re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------------

def wiki_candidates(mwqa: pd.DataFrame, native_titles: list[str], n: int, seed: int = SEED,
                    max_chars: int = MAX_CONTEXT_CHARS, min_chars: int = 400) -> list[dict]:
    """Up to ``n`` contexts from the reserved articles. Every article gets one context before any
    article gets a second, so the set is spread over as many articles as possible."""
    rows = mwqa[mwqa["title"].isin(set(native_titles))]
    order = list(rows.index)
    random.Random(seed).shuffle(order)
    per_title, seen, ranked = {}, set(), []
    for idx in order:
        row = rows.loc[idx]
        lines = A.context_window(row["context"], int(row["answer_start"]), max_chars)
        text = "\n".join(lines)
        if row["answer_text"] not in text or len(text) < min_chars or text in seen:
            continue
        seen.add(text)
        rank = per_title.get(row["title"], 0)
        per_title[row["title"]] = rank + 1
        ranked.append((rank, len(ranked), {
            "id": f"wiki-{idx:05d}", "domain": "wikipedia", "source": row["title"],
            "source_question": row["question"].strip(), "source_answer": row["answer_text"],
            "passages": A.split_passages(lines)}))
    ranked.sort(key=lambda r: r[:2])
    return [r[2] for r in ranked[:n]]


def load_legal(cache: str | Path | None = None) -> pd.DataFrame:
    if cache and Path(cache).exists():
        return pd.read_parquet(cache)
    from datasets import load_dataset

    name, config, split = LEGAL_DATASET
    df = load_dataset(name, config, split=split).to_pandas()
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df


def legal_candidates(legal: pd.DataFrame, n: int, seed: int = SEED, max_chars: int = MAX_CONTEXT_CHARS,
                     min_chars: int = 300) -> list[dict]:
    """Up to ``n`` law articles, one per context, taken round-robin over the laws so no single
    law dominates."""
    ok = legal["is_active"].eq(True) & legal["score_valid"].eq(True) & legal["source_origin"].eq(LEGAL_SOURCE)
    df = legal[ok & legal["context"].str.len().between(min_chars, max_chars)]
    df = df.drop_duplicates("context_key")
    rng = random.Random(seed)
    groups = {}
    for idx in df.index:
        law = df.at[idx, "kanun_adi"]
        groups.setdefault(str(df.at[idx, "kaynak"] if pd.isna(law) else law), []).append(idx)
    laws = sorted(groups)
    rng.shuffle(laws)
    for law in laws:
        rng.shuffle(groups[law])
    out = []
    while len(out) < n and any(groups.values()):
        for law in laws:
            if groups[law] and len(out) < n:
                idx = groups[law].pop()
                row = df.loc[idx]
                lines = [line.strip() for line in row["context"].split("\n") if line.strip()]
                out.append({"id": f"legal-{idx:04d}", "domain": "legal", "source": law,
                            "source_question": row["soru"].strip(), "source_answer": row["cevap"],
                            "passages": A.split_passages(lines)})
    return out


def assign_cells(cands: list[dict], seed: int = SEED) -> list[dict]:
    """Question type, generator and prompt wording per candidate. Each run of four consecutive
    candidates covers all four cells, so any prefix of the list is nearly balanced."""
    rng = random.Random(seed)
    out = []
    for i in range(0, len(cands), len(CELLS)):
        block = list(CELLS)
        rng.shuffle(block)
        for cand, (qtype, gen) in zip(cands[i:i + len(CELLS)], block):
            out.append({**cand, "question_type": qtype, "generator": gen, "variant": rng.randrange(4)})
    return out


# --------------------------------------------------------------------------------------------
# Rewritten ("partial") questions
# --------------------------------------------------------------------------------------------

REWRITE_SYSTEM = """You write test questions for a study of hallucination in Turkish retrieval-augmented generation (RAG).

You receive PASSAGES in Turkish and an ORIGINAL QUESTION about them. Write one new Turkish question about the same subject that the passages answer only PARTLY:
- Its main part must be clearly answerable from the passages.
- It also asks for one specific detail that the passages do not contain but that a reader could plausibly expect them to contain: for example a date, number, name, place, reason or consequence. The detail must really be absent; check the passages before you choose it.
- It reads like a natural question from a real user: one sentence ending with "?", at most 30 words. It does not mention the passages, the text, the article or any source.
- It is not a yes/no question and not a copy of the original question.

Return only json: {"question": "...", "answered_part": "what the passages answer, a few words", "missing_detail": "what they do not contain, a few words"}"""


def rewrite_messages(cand: dict) -> list[dict]:
    return [{"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"PASSAGES:\n{context_text(cand['passages'])}\n\n"
                                        f"ORIGINAL QUESTION: {cand['source_question']}"}]


def _norm(text: str) -> str:
    return " ".join(w.lower() for w in words(text))


def check_rewrite(raw: str, original: str) -> tuple[dict | None, list[str]]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None, ["bad_json"]
    q = str(payload.get("question") or "").strip() if isinstance(payload, dict) else ""
    if not q:
        return None, ["no_question"]
    reasons = []
    n = len(words(q))
    if not q.endswith("?"):
        reasons.append("not_a_question")
    if n < 4 or n > 30:
        reasons.append("length")
    if _norm(q) == _norm(original):
        reasons.append("same_as_original")
    if _META_RE.search(q):
        reasons.append("mentions_source")
    if _english_share(q) > 0.2:
        reasons.append("not_turkish")
    if not str(payload.get("missing_detail") or "").strip():
        reasons.append("no_missing_detail")
    if reasons:
        return None, reasons
    return {"question": q, "answered_part": str(payload.get("answered_part") or "").strip(),
            "missing_detail": str(payload["missing_detail"]).strip()}, []


def with_questions(cands: list[dict], rewrites: dict[str, dict]) -> list[dict]:
    """Final question and detector prompt per candidate. ``rewrites`` maps id → accepted rewrite
    line; partial candidates without an accepted rewrite are dropped."""
    out = []
    for c in cands:
        if c["question_type"] == "partial":
            line = rewrites.get(c["id"])
            if not line or not line["ok"]:
                continue
            extra = line["result"]
            question = extra["question"]
        else:
            extra, question = {}, c["source_question"]
        out.append({**c, "question": question, "answered_part": extra.get("answered_part", ""),
                    "missing_detail": extra.get("missing_detail", ""),
                    "prompt": A.qa_prompt(question, c["passages"], variant=c["variant"])})
    return out


def apply_review(cands: list[dict], review: dict) -> tuple[list[dict], list[str]]:
    """Drop rewritten questions that Claude rejected when reviewing 05a's output, before any answer
    exists (the missing detail is in fact in the context, or the question makes no sense). Returns
    the kept candidates and the rewritten-question ids nobody has reviewed yet."""
    if review.get("rewrite_version") != REWRITE_VERSION:
        raise ValueError(f"review is for rewrite v{review.get('rewrite_version')}, candidates use v{REWRITE_VERSION}")
    reviewed = set(review["reviewed_ids"]) | set(review["rejected"])
    unreviewed = [c["id"] for c in cands if c["question_type"] == "partial" and c["id"] not in reviewed]
    kept = [c for c in cands if not (c["question_type"] == "partial" and c["id"] in review["rejected"])]
    return kept, unreviewed


# --------------------------------------------------------------------------------------------
# Answer checks (objective only: nothing about whether the answer is faithful)
# --------------------------------------------------------------------------------------------

def check_answer(answer: str, finish: str | None = None) -> list[str]:
    a = (answer or "").strip()
    n = len(words(a))
    reasons = []
    if finish == "length":
        reasons.append("max_tokens")
    if "<think>" in a or "</think>" in a:
        reasons.append("thinking_text")
    if n < 3:
        reasons.append("too_short")
    elif _REFUSAL_RE.search(a) and n < 20:
        reasons.append("refusal")
    if n > 250:
        reasons.append("too_long")
    if a and not ends_like_sentence(a):
        reasons.append("unfinished")
    if _english_share(a) > 0.2:
        reasons.append("not_turkish")
    if re.search(r"^\s*#", a, re.MULTILINE):
        reasons.append("markdown_heading")
    return reasons


def answer_messages(prompt: str) -> list[dict]:
    """RAGTruth sent its prompt as the only user message; we do the same for both generators."""
    return [{"role": "user", "content": prompt}]


# --------------------------------------------------------------------------------------------
# Resumable call logs: one JSONL line per attempt, the last line per id wins
# --------------------------------------------------------------------------------------------

def latest(lines: list[dict]) -> dict[str, dict]:
    out = {}
    for line in lines:
        out[line["id"]] = line
    return out


def pending(jobs: list[dict], lines: list[dict], max_attempts: int) -> list[tuple[dict, int]]:
    """Jobs still to run, with their attempt number."""
    last = latest(lines)
    todo = []
    for job in jobs:
        line = last.get(job["id"])
        if line is None:
            todo.append((job, 1))
        elif not line["ok"] and line["attempt"] < max_attempts:
            todo.append((job, line["attempt"] + 1))
    return todo


def _append(path: Path, line: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


async def run_api(client, jobs: list[dict], out_path: str | Path, check, json_mode: bool = False,
                  concurrency: int = 10, budget_usd: float = 1.0, max_attempts: int = 2,
                  max_tokens: int = 800, temperature: float = DEEPSEEK_TEMPERATURE,
                  net_retries: int = 5) -> dict:
    """DeepSeek calls for ``jobs`` ({"id", "messages", ...}); ``check(job, text, finish)`` returns
    (result, reasons). Failed jobs are retried up to ``max_attempts`` in total, also across reruns.
    Stops starting calls once ``budget_usd`` (peak prices, including earlier lines) is spent."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spent = sum(l.get("cost_usd", 0.0) for l in A.read_jsonl(out_path))
    state = {"spent": spent, "calls": 0, "stopped_for_budget": False}
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

    async def call(messages):
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(net_retries):
            try:
                resp = await client.chat.completions.create(
                    model=A.MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}}, **kwargs)
                usage = resp.usage.model_dump() if resp.usage else {}
                return resp.choices[0].message.content or "", usage, resp.choices[0].finish_reason
            except Exception:  # network, rate limit, server errors
                if attempt == net_retries - 1:
                    raise
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))

    async def one(job, attempt):
        async with sem:
            if state["spent"] >= budget_usd:
                state["stopped_for_budget"] = True
                return
            try:
                text, usage, finish = await call(job["messages"])
                result, reasons = check(job, text, finish)
            except Exception as exc:
                text, usage, finish, result = "", {}, None, None
                reasons = [f"api_error: {type(exc).__name__}"]
            cost = A.call_cost(usage) if usage else 0.0
            line = {"id": job["id"], "attempt": attempt, "ok": not reasons, "text": text, "result": result,
                    "reasons": reasons, "finish": finish, "usage": usage, "cost_usd": cost,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
            async with lock:
                _append(out_path, line)
                state["spent"] += cost
                state["calls"] += 1

    for _ in range(max_attempts):
        todo = pending(jobs, A.read_jsonl(out_path), max_attempts)
        if not todo or state["stopped_for_budget"]:
            break
        await asyncio.gather(*(one(job, attempt) for job, attempt in todo))
    last = latest(A.read_jsonl(out_path))
    return {"jobs": len(jobs), "ok": sum(bool(last.get(j["id"], {}).get("ok")) for j in jobs),
            "calls_this_run": state["calls"], "spent_usd": round(state["spent"], 4),
            "stopped_for_budget": state["stopped_for_budget"]}


def run_local(generate, jobs: list[dict], out_path: str | Path, check, batch_size: int = 8,
              max_attempts: int = 2) -> dict:
    """Same log format for a local model. ``generate(list_of_messages, attempt)`` returns a list
    of (text, finish) pairs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calls = 0
    for _ in range(max_attempts):
        todo = pending(jobs, A.read_jsonl(out_path), max_attempts)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            outputs = generate([job["messages"] for job, _ in batch], batch[0][1])
            for (job, attempt), (text, finish) in zip(batch, outputs):
                result, reasons = check(job, text, finish)
                _append(out_path, {"id": job["id"], "attempt": attempt, "ok": not reasons, "text": text,
                                   "result": result, "reasons": reasons, "finish": finish,
                                   "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
                calls += 1
    last = latest(A.read_jsonl(out_path))
    return {"jobs": len(jobs), "ok": sum(bool(last.get(j["id"], {}).get("ok")) for j in jobs),
            "calls_this_run": calls}


def answer_check(job: dict, text: str, finish: str | None):
    reasons = check_answer(text, finish)
    return (text.strip() if not reasons else None), reasons


def rewrite_check(job: dict, text: str, finish: str | None):
    if finish == "length":
        return None, ["max_tokens"]
    return check_rewrite(text, job["source_question"])


def qwen_generator(model_id: str = QWEN_MODEL, max_new_tokens: int = 600, seed: int = SEED):
    """Load Qwen3 in bf16 and return a ``generate`` function for ``run_local`` (thinking off)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    counter = {"batch": 0}

    def generate(batch_messages, attempt):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in batch_messages]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        set_seed(seed + 1000 * attempt + counter["batch"])
        counter["batch"] += 1
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 pad_token_id=tok.pad_token_id, **QWEN_SAMPLING)
        results = []
        for row in out[:, enc["input_ids"].shape[1]:]:
            ended = bool((row == tok.eos_token_id).any() or (row == tok.pad_token_id).any())
            results.append((tok.decode(row, skip_special_tokens=True).strip(), "stop" if ended else "length"))
        return results

    return generate


# --------------------------------------------------------------------------------------------
# Final set and labeling sheet
# --------------------------------------------------------------------------------------------

def select_final(cands: list[dict], answers: dict[str, dict], sizes: dict[str, int] = SIZES) -> tuple[list[dict], dict]:
    """Per domain and cell, the first candidates (in candidate order) with an accepted answer."""
    chosen, shortfall = set(), {}
    for domain, size in sizes.items():
        per_cell = size // len(CELLS)
        for qtype, gen in CELLS:
            pool = [c["id"] for c in cands if c["domain"] == domain and c["question_type"] == qtype
                    and c["generator"] == gen and answers.get(c["id"], {}).get("ok")]
            chosen.update(pool[:per_cell])
            if len(pool) < per_cell:
                shortfall[f"{domain}/{qtype}/{gen}"] = per_cell - len(pool)
    final = [{**c, "answer": answers[c["id"]]["result"]} for c in cands if c["id"] in chosen]
    return final, shortfall


def context_text(passages: list[str]) -> str:
    # "Pasaj n:" rather than "[n]": law texts carry their own "[94]"-style footnote marks.
    return "\n\n".join(f"Pasaj {i}: {p}" for i, p in enumerate(passages, 1))


LABEL_COLUMNS = ("no", "id", "question", "context", "answer", "label", "unsure", "note")


def labeling_sheet(final: list[dict], seed: int = SEED) -> pd.DataFrame:
    """The sheet the user labels: shuffled, and without the generator or question type."""
    order = list(range(len(final)))
    random.Random(seed).shuffle(order)
    rows = [{"no": k + 1, "id": final[i]["id"], "question": final[i]["question"],
             "context": context_text(final[i]["passages"]), "answer": final[i]["answer"],
             "label": "", "unsure": "", "note": ""} for k, i in enumerate(order)]
    return pd.DataFrame(rows, columns=list(LABEL_COLUMNS))


def read_label_sheet(sheet: pd.DataFrame, ids: list[str]) -> tuple[dict[str, int], list[str]]:
    """Labels from a filled-in sheet (1 = hallucinated, 0 = faithful) and a list of problems."""
    problems, labels = [], {}
    sheet = sheet.copy()
    sheet["id"] = sheet["id"].astype(str).str.strip()
    missing = sorted(set(ids) - set(sheet["id"]))
    extra = sorted(set(sheet["id"]) - set(ids))
    if missing:
        problems.append(f"missing ids: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        problems.append(f"unknown ids: {extra[:10]}")
    dup = sheet["id"][sheet["id"].duplicated()].tolist()
    if dup:
        problems.append(f"duplicate ids: {dup[:10]}")
    for row in sheet.itertuples(index=False):
        raw = str(row.label).strip()
        if raw.endswith(".0"):
            raw = raw[:-2]
        if raw in ("0", "1"):
            labels[row.id] = int(raw)
        else:
            problems.append(f"row {row.no} ({row.id}): label must be 0 or 1, got {raw!r}")
    return labels, problems
