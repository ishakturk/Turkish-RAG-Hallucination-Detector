import numpy as np

from hdtr.postprocess import char_probs_to_spans

ANSWER = "Son başvuru 25 Ocak'a kadar yapılabilir."


def _probs(ranges, value=0.9):
    probs = np.zeros(len(ANSWER), dtype=np.float32)
    for s, e in ranges:
        probs[s:e] = value
    return probs


def test_partial_suffix_snaps_to_whole_word():
    # Model flags only "'a" of "Ocak'a" and "25"; the span must cover "25 Ocak'a".
    probs = _probs([(12, 14), (19, 21)])
    spans = char_probs_to_spans(ANSWER, probs, threshold=0.5)
    assert [s["text"] for s in spans] == ["25 Ocak'a"]


def test_short_word_bridges_two_flagged_words():
    text = "Ücret 450 ve 600 TL olarak belirlendi."
    probs = np.zeros(len(text), dtype=np.float32)
    probs[6:9] = 0.8   # 450
    probs[13:16] = 0.8  # 600
    spans = char_probs_to_spans(text, probs, threshold=0.5)
    assert [s["text"] for s in spans] == ["450 ve 600"]


def test_below_threshold_gives_no_span():
    assert char_probs_to_spans(ANSWER, _probs([(12, 21)], 0.3), threshold=0.5) == []


def test_min_chars_drops_single_character_words():
    text = "A ve B"
    probs = np.zeros(len(text), dtype=np.float32)
    probs[0:1] = 0.9
    assert char_probs_to_spans(text, probs, threshold=0.5) == []
