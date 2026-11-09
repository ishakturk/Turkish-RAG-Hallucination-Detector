"""Native Turkish augmentation data generated with DeepSeek (docs/technical-roadmap.md §2.6).

Each multi-wiki-qa row becomes one RAGTruth-style prompt (QA with three passages, or a summary
request), and one API call returns a faithful answer plus, for half of the rows, a copy with
1-3 inserted errors wrapped in <HAL>...</HAL>. The tags become character spans; rows that fail
validation are rejected and counted.

The prompts copy the RAGTruth-TR templates so the detector sees the same input format at
training time. The generator gets the same prompt the detector will see.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .data import _english_share, numbers
from .text import ends_like_sentence, words

MODEL = "deepseek-flash"
# Bump when SYSTEM_PROMPT changes: output files are named by it, so a new gate never reuses old rows.
# v2 (9 Nov): errors inside sentences instead of appended final sentences, tight tags, no arithmetic.
PROMPT_VERSION = 2
BASE_URL = "https://api.deepseek.com"

# USD per 1M tokens, deepseek-flash (verified 2026-11-09). Off-peak is half price.
PRICES = {"hit": 0.006, "miss": 0.30, "output": 1.20}

ERROR_TYPES = ("unsupported", "entity", "number", "contradiction")
ERROR_WEIGHTS = (0.4, 0.2, 0.2, 0.2)  # RAGTruth has more baseless than conflict spans
ERROR_LABELS = {"unsupported": "Evident Baseless Info", "entity": "Evident Conflict",
                "number": "Evident Conflict", "contradiction": "Evident Conflict"}

_TRAILING_SECTIONS = {"kaynakça", "kaynaklar", "dış bağlantılar", "ayrıca bakınız", "notlar",
                      "dipnotlar", "ek okumalar", "konuyla ilgili yayınlar"}
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")
_TURKISH_CHARS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TAG_RE = re.compile(r"</?HAL>")


# --------------------------------------------------------------------------------------------
# Source data
# --------------------------------------------------------------------------------------------

def load_mwqa_tr(cache: str | Path | None = None) -> pd.DataFrame:
    """alexandrainst/multi-wiki-qa, config ``tr`` (5,006 rows, 673 articles), with answer_start."""
    if cache and Path(cache).exists():
        df = pd.read_parquet(cache)
    else:
        from datasets import load_dataset

        df = load_dataset("alexandrainst/multi-wiki-qa", "tr", split="train").to_pandas()
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
    df = df.copy()
    df["answer_start"] = df["answers"].map(lambda a: int(list(a["answer_start"])[0]))
    df["answer_text"] = df["answers"].map(lambda a: str(list(a["text"])[0]))
    return df.reset_index(drop=True)


def split_titles(df: pd.DataFrame, reserve_rows: int = 1000, seed: int = 482) -> dict:
    """Split by article title: ~``reserve_rows`` rows for the native test set, the rest for
    augmentation. The two never share a title."""
    titles = sorted(df["title"].unique())
    random.Random(seed).shuffle(titles)
    counts = df["title"].value_counts()
    native, n = [], 0
    for t in titles:
        if n >= reserve_rows:
            break
        native.append(t)
        n += int(counts[t])
    native_set = set(native)
    return {"seed": seed, "native_titles": sorted(native),
            "augment_titles": sorted(t for t in titles if t not in native_set)}


# --------------------------------------------------------------------------------------------
# Context window and RAGTruth-style prompts
# --------------------------------------------------------------------------------------------

def _is_heading(line: str) -> bool:
    return len(line) < 60 and not ends_like_sentence(line)


def clean_context(text: str, keep_until: int = 0) -> str:
    """Cut trailing Wikipedia sections (references, external links, ...) when they start after
    ``keep_until``."""
    pos = 0
    for line in text.split("\n"):
        if line.strip().lower() in _TRAILING_SECTIONS and pos > keep_until:
            return text[:pos].rstrip()
        pos += len(line) + 1
    return text.rstrip()


def _units(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Non-empty lines with their start offsets; lines longer than ``max_chars`` are split
    into sentences."""
    out, pos = [], 0
    for line in text.split("\n"):
        if line.strip():
            if len(line) > max_chars:
                start = 0
                for sent in _SENT_SPLIT_RE.split(line):
                    idx = line.index(sent, start)
                    out.append((pos + idx, sent))
                    start = idx + len(sent)
            else:
                out.append((pos, line))
        pos += len(line) + 1
    return out


def context_window(context: str, answer_start: int, max_chars: int = 2500) -> list[str]:
    """Lines around the answer, about ``max_chars`` long, so the answer is always included
    and the cost per call stays bounded."""
    text = clean_context(context, keep_until=answer_start)
    units = _units(text, max_chars)
    if not units:
        return []
    centre = max(i for i, (start, _) in enumerate(units) if start <= max(answer_start, units[0][0]))
    lo = hi = centre
    total = len(units[centre][1])
    while True:
        grew = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(units) and not (lo <= j <= hi) and total + len(units[j][1]) <= max_chars:
                total += len(units[j][1])
                lo, hi = min(lo, j), max(hi, j)
                grew = True
        if not grew:
            break
    lines = [u for _, u in units[lo:hi + 1]]
    while len(lines) > 1 and _is_heading(lines[-1]):
        lines.pop()
    return lines


def split_passages(lines: list[str], n: int = 3) -> list[str]:
    """Group lines into ``n`` contiguous passages of similar length (RAGTruth QA uses three)."""
    if len(lines) < n:
        lines = [s for line in lines for s in _SENT_SPLIT_RE.split(line) if s.strip()]
    if len(lines) <= n:
        return list(lines)
    total = sum(len(x) for x in lines)
    groups, current, size = [], [], 0
    for i, line in enumerate(lines):
        current.append(line)
        size += len(line)
        remaining_lines = len(lines) - i - 1
        remaining_groups = n - len(groups) - 1
        if remaining_groups and (size >= total * (len(groups) + 1) / n or remaining_lines == remaining_groups):
            groups.append("\n".join(current))
            current = []
    groups.append("\n".join(current))
    return [g for g in groups if g]


_COUNT_WORDS = {1: "bir", 2: "iki", 3: "üç"}


def qa_prompt(question: str, passages: list[str], variant: int = 0) -> str:
    """RAGTruth-TR QA template. ``variant`` picks among the wordings seen in RAGTruth-TR."""
    verb = "yanıtlayın" if variant & 1 else "cevaplayın"
    ending = "output:" if variant & 2 else "çıktı:"
    body = "\n\n".join(f"pasaj {i}:{p}" for i, p in enumerate(passages, 1))
    return (f"Aşağıdaki soruyu kısaca {verb}:\n{question}\n"
            f"Cevabınızın kesinlikle aşağıdaki {_COUNT_WORDS.get(len(passages), len(passages))} "
            f"pasajı temel alması gerektiğini unutmayın:\n{body}\n\n"
            "Pasajlar soruyu cevaplamak için gerekli bilgileri içermiyorsa, lütfen şu şekilde yanıtlayın: "
            f"\"Verilen pasajlara göre cevap verilemiyor.\"\n{ending}")


def summary_prompt(text: str, n_words: int, variant: int = 0) -> str:
    """RAGTruth-TR Summary template (it says "haberi"/news; kept for format consistency)."""
    head = f"Aşağıdaki haberi {n_words} kelime içinde özetleyin:" if variant & 1 \
        else f"Aşağıdaki haberi {n_words} kelimeyle özetleyin:"
    return f"{head}\n{text}" + (" output:" if variant & 2 else "")


# --------------------------------------------------------------------------------------------
# Plans: what each row asks for, fixed in advance by a seed
# --------------------------------------------------------------------------------------------

def make_plans(rows: pd.DataFrame, seed: int = 482, summary_share: float = 1 / 3,
               hallucinate_share: float = 0.5, max_chars: int = 2500,
               forced: list[tuple[str, bool]] | None = None, min_chars: int = 400) -> list[dict]:
    """One plan per row: task, prompt, whether to hallucinate, error count and types.
    ``forced`` gives (task, hallucinate) per row, used to balance the small quality gate.
    Rows are skipped when the window loses the answer or is shorter than ``min_chars`` (answers
    inside filmography or category lists); the plan id is the row's index in multi-wiki-qa,
    because the dataset's own ``id`` is the article URL and repeats."""
    rng = random.Random(seed)
    plans = []
    for idx, row in zip(rows.index, rows.itertuples(index=False)):
        lines = context_window(row.context, row.answer_start, max_chars)
        joined = "\n".join(lines)
        if row.answer_text not in joined or len(joined) < min_chars:
            continue
        if forced:
            task, hallucinate = forced[len(plans) % len(forced)]
        else:
            task = "Summary" if rng.random() < summary_share else "QA"
            hallucinate = rng.random() < hallucinate_share
        variant = rng.randrange(4)
        if task == "QA":
            prompt = qa_prompt(row.question, split_passages(lines), variant=variant)
            spec = {"task": "QA"}
        else:
            text = "\n".join(lines)
            n_words = int(min(max(50, rng.randint(60, 130)), max(50, len(words(text)) // 2)))
            prompt = summary_prompt(text, n_words, variant=variant)
            spec = {"task": "Summary", "summary_words": n_words}
        n_errors = rng.choices((1, 2, 3), (0.5, 0.35, 0.15))[0] if hallucinate else 0
        spec.update(hallucinate=hallucinate, n_errors=n_errors,
                    error_types=rng.choices(ERROR_TYPES, ERROR_WEIGHTS, k=n_errors))
        plans.append({"id": f"mwqa{idx:05d}", "row": int(idx), "title": row.title, "task_type": task,
                      "prompt": prompt, "spec": spec})
    return plans


# --------------------------------------------------------------------------------------------
# Generator messages
# --------------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You create training data for a Turkish hallucination detector for retrieval-augmented generation (RAG).

You receive a TASK PROMPT in Turkish (a question with passages, or a text to summarize) and a SPEC.

Step 1, "faithful": respond to the TASK PROMPT in natural, fluent Turkish, the way a careful AI assistant would.
- Use only information stated in the passages or text. Do not add background knowledge, even if it is true.
- Do not compute intervals, totals or counts that the text does not state ("two years later" when the text gives 2000 and 2003); reuse the text's own figures and wording.
- QA: 2-5 sentences that answer the question. Summary: about SPEC.summary_words words.
- Plain text only: no Markdown headings, no bullet lists, no <HAL> tags.

Step 2, "hallucinated" (only when SPEC.hallucinate is true, otherwise null): copy the faithful response and introduce exactly SPEC.n_errors errors, one per type listed in SPEC.error_types, changing as little else as possible.
- entity: replace a name, place, organization or title with a different, plausible one.
- number: change a number, date, quantity or ranking.
- contradiction: make a statement say the opposite of, or something incompatible with, the context.
- unsupported: add a plausible-sounding claim that the context does not state. In most cases work it into an existing sentence as a clause, modifier or detail (a reason, date, place, name, number or consequence), anywhere in the response. Only occasionally add it as a separate sentence, and then not always as the last one. Prefer specific details over vague praise such as "büyük ilgi gördü" or "büyük başarı elde etti".
Every error must be detectable by a careful reader who has only the context. Spread errors over the response; do not always put them at the end. Keep errors fluent and grammatical: adjust Turkish suffixes so the sentence reads naturally, and keep the rest of the response consistent with the error (do not leave a sentence that contradicts it).
Wrap exactly the changed or added words of each error in <HAL>...</HAL>. Keep tags tight: only the wrong words with their suffixes, never the correct words around them (if you add "Ankara'daki" before a correct school name, tag only "Ankara'daki"). Never nest tags, and tag nothing else.

"errors": one item per <HAL> tag, in the same order: {"type": the error type, "original": the faithful words it replaced ("" for unsupported), "changed": the words inside the tag}.

Return only json in this format:
{"faithful": "Dizi 2011'de ilk kez yayınlandı ve üç sezon sürdü.", "hallucinated": "Dizi <HAL>2013'te</HAL> ilk kez yayınlandı ve üç sezon sürdü.", "errors": [{"type": "number", "original": "2011'de", "changed": "2013'te"}]}
With SPEC.hallucinate false: {"faithful": "...", "hallucinated": null, "errors": []}"""


def build_messages(plan: dict) -> list[dict]:
    spec = json.dumps(plan["spec"], ensure_ascii=False)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SPEC: {spec}\n\nTASK PROMPT:\n<<<\n{plan['prompt']}\n>>>"}]


# --------------------------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------------------------

def parse_tags(text: str) -> tuple[str, list[dict]] | None:
    """Remove <HAL> tags; return the clean text and character spans, or None if the tags are
    unbalanced, nested or empty."""
    clean, spans, pos, open_at = [], [], 0, None
    out_len = 0
    for m in _TAG_RE.finditer(text):
        piece = text[pos:m.start()]
        clean.append(piece)
        out_len += len(piece)
        if m.group() == "<HAL>":
            if open_at is not None:
                return None
            open_at = out_len
        else:
            if open_at is None:
                return None
            spans.append({"start": open_at, "end": out_len})
            open_at = None
        pos = m.end()
    if open_at is not None:
        return None
    clean.append(text[pos:])
    clean_text = "".join(clean)
    trimmed = []
    for s in spans:
        start, end = s["start"], s["end"]
        while start < end and clean_text[start].isspace():
            start += 1
        while end > start and clean_text[end - 1].isspace():
            end -= 1
        if start == end:
            return None
        trimmed.append({"start": start, "end": end})
    return clean_text, trimmed


def _is_turkish(text: str) -> bool:
    return bool(_TURKISH_CHARS_RE.search(text)) and _english_share(text) < 0.15


def _check_response(text: str, prompt: str, task: str) -> list[str]:
    reasons = []
    if not isinstance(text, str) or len(text.strip()) < 40:
        return ["too_short"]
    if "<" in text and re.search(r"</?\s*hal", text, re.I):
        reasons.append("stray_tag")
    if not _is_turkish(text):
        reasons.append("not_turkish")
    if not ends_like_sentence(text.strip()):
        reasons.append("truncated")
    if task == "QA" and "cevap verilemiyor" in text:
        reasons.append("refusal")
    return reasons


def validate(plan: dict, payload: dict) -> tuple[list[dict], list[str]]:
    """Turn one API response into training records (faithful, plus hallucinated if planned).
    Returns (records, rejection reasons); any reason rejects the whole row."""
    if not isinstance(payload, dict) or not isinstance(payload.get("faithful"), str):
        return [], ["bad_json"]
    prompt, task, spec = plan["prompt"], plan["task_type"], plan["spec"]
    faithful = payload["faithful"].strip()
    reasons = _check_response(faithful, prompt, task)
    if reasons:
        return [], [f"faithful_{r}" for r in reasons]
    if numbers(faithful) - numbers(prompt):
        reasons.append("faithful_number_not_in_context")

    base = {"source_id": f"mwqa:{plan['title']}", "aug_id": plan["id"], "prompt": prompt,
            "task_type": task, "dataset": "multi-wiki-qa-aug", "language": "tr", "model": MODEL}
    records = [{**base, "variant": "faithful", "answer": faithful, "labels": [], "errors": []}]

    if spec["hallucinate"]:
        raw = payload.get("hallucinated")
        errors = payload.get("errors") or []
        parsed = parse_tags(raw.strip()) if isinstance(raw, str) else None
        if parsed is None:
            return [], reasons + ["hallucinated_bad_tags"]
        text, spans = parsed
        reasons += [f"hallucinated_{r}" for r in _check_response(text, prompt, task)]
        if not spans:
            reasons.append("hallucinated_no_tags")
        elif len(spans) != len(errors) or len(spans) > 3:
            reasons.append("hallucinated_error_count")
        if text == faithful:
            reasons.append("hallucinated_no_change")
        elif SequenceMatcher(None, faithful, text, autojunk=False).ratio() < 0.7:
            reasons.append("hallucinated_rewritten")
        for s in spans:
            span_text = text[s["start"]:s["end"]]
            if len(words(span_text)) >= 4 and span_text in prompt:
                reasons.append("hallucinated_span_in_context")
                break
        if reasons:
            return [], reasons
        types = [e.get("type") if isinstance(e, dict) else None for e in errors]
        labels = [{**s, "label": ERROR_LABELS.get(t, "Unknown")} for s, t in zip(spans, types)]
        records.append({**base, "variant": "hallucinated", "answer": text, "labels": labels,
                        "errors": errors})
    elif reasons:
        return [], reasons
    return records, []


# --------------------------------------------------------------------------------------------
# Cost and peak hours
# --------------------------------------------------------------------------------------------

def call_cost(usage: dict, off_peak: bool = False) -> float:
    """USD for one call from DeepSeek's usage fields."""
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = (usage.get("prompt_tokens", 0) or 0) - hit
    out = usage.get("completion_tokens", 0) or 0
    cost = (hit * PRICES["hit"] + miss * PRICES["miss"] + out * PRICES["output"]) / 1e6
    return cost / 2 if off_peak else cost


def is_peak(utc: time.struct_time | None = None) -> bool:
    """DeepSeek peak pricing: weekdays 01:00-04:00 and 06:00-10:00 UTC."""
    t = utc or time.gmtime()
    return t.tm_wday < 5 and (1 <= t.tm_hour < 4 or 6 <= t.tm_hour < 10)


# --------------------------------------------------------------------------------------------
# Batch runner (resumable JSONL)
# --------------------------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _call(client, plan: dict, max_tokens: int, temperature: float, net_retries: int = 5):
    for attempt in range(net_retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL, messages=build_messages(plan), temperature=temperature,
                max_tokens=max_tokens, response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}})
            usage = resp.usage.model_dump() if resp.usage else {}
            return resp.choices[0].message.content or "", usage, resp.choices[0].finish_reason
        except Exception as exc:  # network, rate limit, server errors
            if attempt == net_retries - 1:
                raise
            await asyncio.sleep(min(60, 2 ** attempt + random.random()))


async def run_batch(client, plans: list[dict], out_path: str | Path, concurrency: int = 20,
                    budget_usd: float = 4.0, max_attempts: int = 2, max_tokens: int = 1500,
                    temperature: float = 0.7, progress_every: int = 100) -> dict:
    """Generate all plans not yet in ``out_path``; one JSONL line per plan with the records,
    rejection reasons, usage and cost. Stops starting new calls once ``budget_usd`` (counted
    at peak prices, including earlier lines in the file) is spent."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(out_path)
    done_ids = {d["id"] for d in done}
    spent = sum(d.get("cost_usd", 0.0) for d in done)
    todo = [p for p in plans if p["id"] not in done_ids]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    state = {"spent": spent, "written": 0, "stopped_for_budget": False}

    async def one(plan):
        async with sem:
            if state["spent"] >= budget_usd:
                state["stopped_for_budget"] = True
                return
            usages, history, records, raw, finish = [], [], [], "", None
            t0 = time.perf_counter()
            for _ in range(max_attempts):
                try:
                    raw, usage, finish = await _call(client, plan, max_tokens, temperature)
                except Exception as exc:
                    history.append([f"api_error: {type(exc).__name__}: {exc}"[:200]])
                    break
                usages.append(usage)
                try:
                    payload = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    payload = None
                records, reasons = validate(plan, payload)
                if finish == "length":
                    records, reasons = [], ["max_tokens"]
                history.append(reasons)
                if records:
                    break
            cost = sum(call_cost(u) for u in usages)
            line = {"id": plan["id"], "task_type": plan["task_type"], "spec": plan["spec"],
                    "accepted": bool(records), "records": records, "reasons": history,
                    "raw": raw, "usage": usages, "cost_usd": cost, "attempts": len(usages),
                    "sec": round(time.perf_counter() - t0, 2),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                state["spent"] += cost
                state["written"] += 1
                if progress_every and state["written"] % progress_every == 0:
                    print(f"{state['written']}/{len(todo)} done, ${state['spent']:.3f} spent", flush=True)

    await asyncio.gather(*(one(p) for p in todo))
    return {"planned": len(plans), "already_done": len(done_ids), "written": state["written"],
            "spent_usd": state["spent"], "stopped_for_budget": state["stopped_for_budget"]}


# --------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------

def records_frame(lines: list[dict]) -> pd.DataFrame:
    rows = [r for line in lines if line.get("accepted") for r in line["records"]]
    return pd.DataFrame(rows)


def _final_sentence(answer: str, span: dict) -> bool:
    """The span is a whole sentence at the very end of the answer."""
    before = answer[:span["start"]].rstrip()
    return len(answer.rstrip()) - span["end"] <= 2 and (not before or before[-1] in ".!?:")


def batch_stats(lines: list[dict]) -> dict:
    from collections import Counter

    n = len(lines)
    accepted = [l for l in lines if l.get("accepted")]
    first_try = sum(1 for l in accepted if l["attempts"] == 1)
    reasons = Counter(r for l in lines for attempt in l["reasons"] for r in attempt)
    usage = [u for l in lines for u in l["usage"]]
    cost = sum(call_cost(u) for u in usage)
    frame = records_frame(lines)
    by = frame.groupby(["task_type", "variant"]).size().to_dict() if len(frame) else {}
    hall = [r for l in accepted for r in l["records"] if r["variant"] == "hallucinated"]
    spans = [len(r["labels"]) for r in hall]
    all_spans = [(r["answer"], s) for r in hall for s in r["labels"]]
    return {
        "rows": n, "accepted_rows": len(accepted),
        "acceptance_rate": round(len(accepted) / n, 3) if n else None,
        "accepted_first_try": first_try, "calls": len(usage),
        "rejection_reasons": dict(reasons.most_common()),
        "records": len(frame), "records_by_task_variant": {f"{k[0]}/{k[1]}": v for k, v in by.items()},
        "spans_per_hallucinated": round(sum(spans) / len(spans), 2) if spans else None,
        # Position shortcut check: RAGTruth-TR QA+Summary has 11% of spans as a whole final sentence.
        "span_final_sentence_share": round(sum(_final_sentence(a, s) for a, s in all_spans) / len(all_spans), 3)
        if all_spans else None,
        "tokens": {k: int(sum((u.get(k) or 0) for u in usage)) for k in
                   ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")},
        "cost_usd_peak": round(cost, 4), "cost_usd_off_peak": round(cost / 2, 4),
        "cost_per_row_peak": round(cost / n, 6) if n else None,
        "median_sec_per_row": float(pd.Series([l["sec"] for l in lines]).median()) if n else None,
    }


def mark_spans(text: str, spans: list[dict], left: str = "[[", right: str = "]]") -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += [text[pos:s["start"]], left, text[s["start"]:s["end"]], right]
        pos = s["end"]
    return "".join(out + [text[pos:]])
