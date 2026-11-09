import numpy as np

from hdtr.baselines import LexicalOverlap, WordLogReg, sentence_spans, tr_lower

PROMPT = "Başvurular 15 Ocak'ta kapanır. Ücret 450 TL'dir. Öğrenciler ücret ödemez."
SUPPORTED = "Başvurular 15 Ocak'ta kapanır."
UNSUPPORTED = "Ücretler yedi iş günü içinde otomatik olarak iade edilir."


def test_tr_lower_handles_dotted_and_dotless_i():
    assert tr_lower("IŞIK İzmir") == "ışık izmir"


def test_sentence_spans():
    text = "Bir cümle. İkinci cümle!"
    assert [text[s:e] for s, e in sentence_spans(text)] == ["Bir cümle.", "İkinci cümle!"]


def test_lexical_overlap_scores_unsupported_sentence_higher():
    lex = LexicalOverlap().fit([PROMPT], [SUPPORTED, UNSUPPORTED])
    answer = f"{SUPPORTED} {UNSUPPORTED}"
    probs = lex.predict_char_probs(PROMPT, answer)
    assert probs[len(SUPPORTED) + 2:].mean() > probs[: len(SUPPORTED)].mean()


def test_word_logreg_learns_out_of_context_words():
    answer = f"{SUPPORTED} {UNSUPPORTED}"
    gold = [{"start": len(SUPPORTED) + 1, "end": len(answer)}]
    lex = LexicalOverlap().fit([PROMPT], [answer])
    clf = WordLogReg(lex).fit([PROMPT] * 20, [answer] * 20, [gold] * 20)
    probs = clf.predict_char_probs(PROMPT, answer)
    assert probs[gold[0]["start"]:].mean() > probs[: len(SUPPORTED)].mean()
    assert set(clf.coefficients()) >= {"in_ctx", "stem_in_ctx"}
    assert np.all((probs >= 0) & (probs <= 1))
