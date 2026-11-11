import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hdtr import native as N

CONTEXT = ("Letonya Futbol Ligi 1927 yılında kuruldu. Lig, ülkenin en üst düzey futbol organizasyonudur. "
           "Skonto FC ligde 14 kez şampiyon olarak rekor kırdı. Maçlar her yıl ilkbahar ve sonbahar arasında oynanır.")


def _cand(i, domain="wikipedia", qtype="original", gen="qwen"):
    return {"id": f"{domain[:4]}-{i}", "domain": domain, "source": "Lig", "source_question": "Lig ne zaman kuruldu?",
            "source_answer": "1927", "passages": [CONTEXT], "question_type": qtype, "generator": gen, "variant": 0}


def test_assign_cells_balances_every_block_of_four_and_is_deterministic():
    cands = [_cand(i) for i in range(10)]
    a, b = N.assign_cells(cands, seed=1), N.assign_cells(cands, seed=1)
    assert [(c["question_type"], c["generator"]) for c in a] == [(c["question_type"], c["generator"]) for c in b]
    for i in (0, 4):
        assert {(c["question_type"], c["generator"]) for c in a[i:i + 4]} == set(N.CELLS)


def test_wiki_candidates_spread_over_titles_before_repeating():
    rows = []
    for t in ("A", "B", "C"):
        for k in range(3):
            ctx = f"{t} başlığı {k}. " + CONTEXT * 2
            rows.append({"title": t, "context": ctx, "answer_start": ctx.index("1927"), "answer_text": "1927",
                         "question": f"{t} {k}?"})
    cands = N.wiki_candidates(pd.DataFrame(rows), ["A", "B"], n=3, min_chars=100)
    assert len(cands) == 3 and {c["source"] for c in cands[:2]} == {"A", "B"}
    assert all(c["id"].startswith("wiki-") and c["passages"] for c in cands)


def test_legal_candidates_round_robin_and_filters():
    body = "Madde 1 – " + "Bu Kanun kamu kurumlarını kapsar. " * 12
    df = pd.DataFrame({
        "kanun_adi": ["K1", "K1", "K1", np.nan, "K2"], "kaynak": ["K1", "K1", "K1", "Tüzük", "K2"],
        "context": [body, body + "x", body + "y", body + "z", body + "w"], "context_key": list("abcde"),
        "soru": ["Soru?"] * 5, "cevap": ["Cevap."] * 5,
        "is_active": [True, True, False, True, True], "score_valid": [True] * 5,
        "source_origin": [N.LEGAL_SOURCE] * 5})
    cands = N.legal_candidates(df, n=3)
    assert len(cands) == 3 and len({c["source"] for c in cands}) == 3   # K1, K2, Tüzük once each
    assert "legal-0002" not in {c["id"] for c in cands}                 # inactive row
    df.loc[4, "source_origin"] = None                                    # templated question: excluded
    assert "legal-0004" not in {c["id"] for c in N.legal_candidates(df, n=5)}


def test_apply_review_drops_rejected_rewrites_and_reports_unreviewed():
    cands = [_cand(0), _cand(1, qtype="partial"), _cand(2, qtype="partial"), _cand(3, qtype="partial")]
    review = {"rewrite_version": N.REWRITE_VERSION, "reviewed_ids": ["wiki-1"], "rejected": {"wiki-2": "detail present"}}
    kept, unreviewed = N.apply_review(cands, review)
    assert [c["id"] for c in kept] == ["wiki-0", "wiki-1", "wiki-3"] and unreviewed == ["wiki-3"]


def test_check_rewrite():
    good = json.dumps({"question": "Letonya Futbol Ligi ne zaman kuruldu ve ilk şampiyonu hangi takımdı?",
                       "answered_part": "kuruluş yılı", "missing_detail": "ilk şampiyon"})
    out, reasons = N.check_rewrite(good, "Lig ne zaman kuruldu?")
    assert reasons == [] and out["missing_detail"] == "ilk şampiyon"
    bad = json.dumps({"question": "Pasajlara göre lig ne zaman kuruldu", "missing_detail": ""})
    _, reasons = N.check_rewrite(bad, "x")
    assert {"not_a_question", "mentions_source", "no_missing_detail"} <= set(reasons)
    assert N.check_rewrite("Lig ne zaman kuruldu?", "x")[1] == ["bad_json"]
    same = json.dumps({"question": "Lig ne zaman kuruldu?", "missing_detail": "y"})
    assert "same_as_original" in N.check_rewrite(same, "lig  ne zaman KURULDU?")[1]
    named = json.dumps({"question": "Metin Oktay hangi kulüpte kaç gol attı?", "missing_detail": "gol"})
    assert N.check_rewrite(named, "x")[1] == []                          # a first name, not "the text"


def test_check_answer():
    assert N.check_answer("Lig 1927 yılında kuruldu.") == []
    assert N.check_answer("Verilen pasajlara göre cevap verilemiyor.") == ["refusal"]
    long_refusal = ("Lig 1927 yılında kuruldu ve Skonto FC 14 kez şampiyon oldu. Ancak ilk şampiyonun "
                    "kim olduğu belirtilmediği için bu kısma cevap verilemiyor.")
    assert N.check_answer(long_refusal) == []                           # partial answer is kept
    assert "unfinished" in N.check_answer("Lig 1927 yılında kuruldu ve")
    assert "max_tokens" in N.check_answer("Lig kuruldu.", "length")
    assert "thinking_text" in N.check_answer("<think>hmm</think> Lig kuruldu.")
    assert "not_turkish" in N.check_answer("The league was founded in 1927 and it is the top level of the sport.")
    assert "markdown_heading" in N.check_answer("# Cevap\nLig 1927 yılında kuruldu.")


def test_with_questions_drops_partial_without_rewrite_and_builds_prompt():
    cands = [_cand(0), _cand(1, qtype="partial"), _cand(2, qtype="partial")]
    rewrites = {"wiki-1": {"ok": True, "result": {"question": "Lig ne zaman ve kim tarafından kuruldu?",
                                                  "answered_part": "yıl", "missing_detail": "kurucu"}},
                "wiki-2": {"ok": False, "result": None}}
    out = N.with_questions(cands, rewrites)
    assert [c["id"] for c in out] == ["wiki-0", "wiki-1"]
    assert out[1]["question"].startswith("Lig ne zaman ve kim") and out[1]["missing_detail"] == "kurucu"
    assert out[0]["prompt"].startswith("Aşağıdaki soruyu kısaca") and "pasaj 1:" in out[0]["prompt"]


class _FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def create(self, **kw):
        assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        usage = SimpleNamespace(model_dump=lambda: {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1000,
                                                    "completion_tokens": 100})
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                                     finish_reason="stop")])


def test_run_api_retries_failures_and_resumes(tmp_path):
    jobs = [{"id": f"j{i}", "messages": N.answer_messages("soru")} for i in range(2)]
    refusal = "Verilen pasajlara göre cevap verilemiyor."
    comp = _FakeCompletions([refusal, refusal, "Lig 1927 yılında kuruldu."])   # round 1 fails, round 2 passes
    client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    out = tmp_path / "a.jsonl"
    s = asyncio.run(N.run_api(client, jobs, out, N.answer_check, concurrency=1))
    assert s["ok"] == 2 and comp.calls == 4                            # each job: refusal, then accepted
    s = asyncio.run(N.run_api(client, jobs, out, N.answer_check))
    assert s["calls_this_run"] == 0 and comp.calls == 4
    last = N.latest(N.A.read_jsonl(out))
    assert last["j0"]["attempt"] == 2 and last["j0"]["result"] == "Lig 1927 yılında kuruldu."


def test_run_local_and_select_final(tmp_path):
    cands = N.assign_cells([_cand(i) for i in range(8)] + [_cand(i, "legal") for i in range(4)], seed=3)
    calls = []

    def generate(batch, attempt):
        calls.append(attempt)
        return [("Lig 1927 yılında kuruldu.", "stop")] * len(batch)

    jobs = [{"id": c["id"], "messages": N.answer_messages("p")} for c in cands]
    s = N.run_local(generate, jobs, tmp_path / "q.jsonl", N.answer_check, batch_size=5)
    assert s["ok"] == 12 and calls == [1, 1, 1]
    answers = N.latest(N.A.read_jsonl(tmp_path / "q.jsonl"))
    del answers[cands[0]["id"]]
    final, short = N.select_final(cands, answers, sizes={"wikipedia": 8, "legal": 4})
    cell0 = f"wikipedia/{cands[0]['question_type']}/{cands[0]['generator']}"
    assert len(final) == 11 and short == {cell0: 1}
    counts = Counter((c["domain"], c["question_type"], c["generator"]) for c in final)
    assert all(v == 1 for k, v in counts.items() if k[0] == "legal")


def test_labeling_sheet_hides_design_and_read_label_sheet_reports_problems():
    final = [{**_cand(i, gen="deepseek"), "question": "Soru?", "answer": "Cevap."} for i in range(4)]
    sheet = N.labeling_sheet(final)
    assert list(sheet.columns) == list(N.LABEL_COLUMNS) and "generator" not in sheet.columns
    assert sorted(sheet["id"]) == [c["id"] for c in final] and sheet["no"].tolist() == [1, 2, 3, 4]
    assert sheet["context"].iloc[0].startswith("Pasaj 1: ")
    sheet["label"] = ["1", "0", 1.0, "evet"]
    labels, problems = N.read_label_sheet(sheet, [c["id"] for c in final] + ["wiki-9"])
    assert len(labels) == 3 and set(labels.values()) == {0, 1}
    assert any("evet" in p for p in problems) and any("wiki-9" in p for p in problems)
