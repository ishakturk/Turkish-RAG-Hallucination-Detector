import asyncio
import json
from types import SimpleNamespace

import pandas as pd

from hdtr import augment as A

CONTEXT = (
    "Letonya'da futbol, Letonya Futbol Federasyonu tarafından yönetilmektedir.\n\n"
    "Tarihçe\n"
    "Letonya'da en üst düzeydeki futbol ligi 1927 yılında kuruldu. 1943 yılında ise bu lig sona erdi.\n\n"
    "Virslīga adı verilen lig 1991'de kuruldu. Skonto FC üst üste 14 şampiyonluk kazandı.\n\n"
    "Kaynakça\n\nDış bağlantılar\n League321.com"
)


def _plan(hallucinate=True, n_errors=1, task="QA"):
    return {"id": "x#QA", "title": "Letonya'da futbol", "task_type": task,
            "prompt": A.qa_prompt("Lig ne zaman kuruldu?", ["Lig 1927 yılında kuruldu.", "Skonto FC 14 kez şampiyon oldu."]),
            "spec": {"task": task, "hallucinate": hallucinate, "n_errors": n_errors, "error_types": ["number"] * n_errors}}


def test_clean_context_drops_reference_sections():
    out = A.clean_context(CONTEXT)
    assert "Kaynakça" not in out and "League321" not in out and out.endswith("şampiyonluk kazandı.")


def test_context_window_keeps_answer_and_respects_budget():
    start = CONTEXT.index("1991")
    lines = A.context_window(CONTEXT, start, max_chars=120)
    joined = "\n".join(lines)
    assert "1991" in joined and len(joined) <= 120 + len(lines)
    assert not A._is_heading(lines[-1])


def test_split_passages_gives_three_contiguous_groups():
    lines = [f"Cümle {i} burada bitiyor." for i in range(7)]
    ps = A.split_passages(lines, 3)
    assert len(ps) == 3 and "\n".join(ps) == "\n".join(lines)
    assert len(A.split_passages(["Bir. İki. Üç. Dört."], 3)) == 3


def test_prompts_follow_ragtruth_templates():
    p = A.qa_prompt("Soru?", ["a", "b", "c"], variant=3)
    assert p.startswith("Aşağıdaki soruyu kısaca yanıtlayın:\nSoru?\n") and "üç pasajı" in p
    assert "pasaj 3:c" in p and p.endswith("output:")
    assert A.summary_prompt("Metin.", 80).startswith("Aşağıdaki haberi 80 kelimeyle özetleyin:\nMetin.")


def test_parse_tags_spans_and_errors():
    text, spans = A.parse_tags("Lig <HAL>1931'de</HAL> kuruldu ve <HAL> 20 </HAL> kez kazandı.")
    assert text == "Lig 1931'de kuruldu ve  20  kez kazandı."
    assert [text[s["start"]:s["end"]] for s in spans] == ["1931'de", "20"]
    assert A.parse_tags("a <HAL>b <HAL>c</HAL></HAL>") is None
    assert A.parse_tags("a </HAL>b") is None
    assert A.parse_tags("a <HAL>b") is None
    assert A.parse_tags("a <HAL> </HAL> b") is None


def test_validate_accepts_good_pair():
    payload = {"faithful": "Letonya'daki lig 1927 yılında kuruldu ve Skonto FC 14 kez şampiyon oldu.",
               "hallucinated": "Letonya'daki lig <HAL>1931</HAL> yılında kuruldu ve Skonto FC 14 kez şampiyon oldu.",
               "errors": [{"type": "number", "original": "1927", "changed": "1931"}]}
    records, reasons = A.validate(_plan(), payload)
    assert reasons == [] and [r["variant"] for r in records] == ["faithful", "hallucinated"]
    h = records[1]
    assert h["labels"] == [{"start": 17, "end": 21, "label": "Evident Conflict"}]
    assert h["answer"][17:21] == "1931" and records[0]["labels"] == []


def test_validate_rejections():
    good = "Letonya'daki lig 1927 yılında kuruldu ve Skonto FC 14 kez şampiyon oldu."
    # number in the faithful answer that the context does not contain
    _, r = A.validate(_plan(hallucinate=False), {"faithful": good.replace("1927", "1925")})
    assert "faithful_number_not_in_context" in r
    # English text
    _, r = A.validate(_plan(hallucinate=False), {"faithful": "The league was founded in 1927 and it is the top league of the country."})
    assert "faithful_not_turkish" in r
    # tag count differs from the error list
    _, r = A.validate(_plan(), {"faithful": good, "hallucinated": good.replace("1927", "<HAL>1931</HAL>"), "errors": []})
    assert "hallucinated_error_count" in r
    # unchanged text inside tags
    _, r = A.validate(_plan(), {"faithful": good, "hallucinated": good.replace("1927", "<HAL>1927</HAL>"),
                                "errors": [{"type": "number"}]})
    assert "hallucinated_no_change" in r
    # rewritten instead of edited
    _, r = A.validate(_plan(), {"faithful": good, "errors": [{"type": "unsupported"}],
                                "hallucinated": "Takım <HAL>Avrupa'da da başarılı oldu</HAL> ve ülkede çok sevilen bir spor haline geldi."})
    assert "hallucinated_rewritten" in r
    # truncated / bad json
    _, r = A.validate(_plan(hallucinate=False), {"faithful": "Letonya'daki lig 1927 yılında kuruldu ve Skonto FC"})
    assert "faithful_truncated" in r
    assert A.validate(_plan(), None) == ([], ["bad_json"])


def test_make_plans_is_deterministic_and_balanced_when_forced():
    long_context = CONTEXT.replace("kuruldu.", "kuruldu. " + "Lig her yıl düzenli olarak oynandı. " * 5)
    rows = pd.DataFrame([{"id": "same-url", "title": "T", "context": long_context, "question": "Lig ne zaman kuruldu?",
                          "answer_start": long_context.index("1927"), "answer_text": "1927"} for i in range(6)])
    short = rows.iloc[:1].assign(context=CONTEXT, answer_start=CONTEXT.index("1927"))
    a, b = A.make_plans(rows, seed=1), A.make_plans(rows, seed=1)
    assert a == b
    forced = A.make_plans(rows, forced=[("QA", True), ("Summary", False)])
    assert [p["task_type"] for p in forced] == ["QA", "Summary"] * 3
    assert all(p["spec"]["n_errors"] >= 1 for p in forced if p["spec"]["hallucinate"])
    assert "1927" in forced[0]["prompt"]
    assert len({p["id"] for p in a}) == 6          # unique although the dataset id repeats
    assert A.make_plans(short) == []                # window below min_chars is skipped


def test_split_titles_disjoint():
    df = pd.DataFrame({"title": [f"t{i % 50}" for i in range(500)]})
    s = A.split_titles(df, reserve_rows=100)
    assert not set(s["native_titles"]) & set(s["augment_titles"])
    assert df["title"].isin(s["native_titles"]).sum() >= 100


def test_call_cost_and_peak():
    usage = {"prompt_cache_hit_tokens": 1_000_000, "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert abs(A.call_cost(usage) - 1.506) < 1e-9 and abs(A.call_cost(usage, off_peak=True) - 0.753) < 1e-9
    import time
    assert A.is_peak(time.strptime("2026-11-09 07:30", "%Y-%m-%d %H:%M"))      # Monday
    assert not A.is_peak(time.strptime("2026-11-08 07:30", "%Y-%m-%d %H:%M"))  # Sunday


class _FakeCompletions:
    def __init__(self, content):
        self.content, self.calls = content, 0

    async def create(self, **kw):
        self.calls += 1
        assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
        usage = SimpleNamespace(model_dump=lambda: {"prompt_cache_hit_tokens": 500,
                                                    "prompt_cache_miss_tokens": 1000, "completion_tokens": 300})
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def test_run_batch_resumes_and_retries(tmp_path):
    good = json.dumps({"faithful": "Letonya'daki lig 1927 yılında kuruldu ve Skonto FC 14 kez şampiyon oldu.",
                       "hallucinated": None, "errors": []})
    comp = _FakeCompletions(good)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    plans = [dict(_plan(hallucinate=False), id=f"p{i}") for i in range(3)]
    out = tmp_path / "gen.jsonl"
    s = asyncio.run(A.run_batch(client, plans, out, concurrency=2, progress_every=0))
    assert s["written"] == 3 and comp.calls == 3
    s = asyncio.run(A.run_batch(client, plans, out, progress_every=0))
    assert s["written"] == 0 and comp.calls == 3
    stats = A.batch_stats(A.read_jsonl(out))
    assert stats["accepted_rows"] == 3 and stats["records"] == 3

    bad = _FakeCompletions("not json")
    client = SimpleNamespace(chat=SimpleNamespace(completions=bad))
    asyncio.run(A.run_batch(client, [dict(_plan(), id="q")], tmp_path / "bad.jsonl", progress_every=0))
    line = A.read_jsonl(tmp_path / "bad.jsonl")[0]
    assert bad.calls == 2 and not line["accepted"] and line["reasons"] == [["bad_json"], ["bad_json"]]
