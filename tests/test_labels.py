import numpy as np
import pytest

from hdtr.labels import IGNORE_INDEX, char_mask, label_tokens, token_probs_to_char_probs

ANSWER = "Son başvuru 25 Ocak'a kadar."  # "25 Ocak'a" is hallucinated
SPAN = [{"start": 12, "end": 21}]


def test_char_mask_covers_span():
    mask = char_mask(len(ANSWER), SPAN)
    assert ANSWER[mask.argmax(): len(mask) - mask[::-1].argmax()] == "25 Ocak'a"


def test_char_mask_clips_out_of_bounds():
    assert char_mask(5, [{"start": 3, "end": 99}]).sum() == 2


def test_label_tokens_masks_prompt_and_specials():
    # Prompt token, [SEP], then answer sub-tokens "Son", " başvuru", " 25", " Ocak", "'a", " kadar", "."
    offsets = [(0, 6), (0, 0), (0, 3), (4, 11), (12, 14), (15, 19), (19, 21), (22, 27), (27, 28)]
    is_answer = [False, False, True, True, True, True, True, True, True]
    labels = label_tokens(offsets, is_answer, SPAN, len(ANSWER))
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 0, 0, 1, 1, 1, 0, 0]


def test_dotted_capital_i_keeps_offsets():
    # Lowercasing would change the length; we never lowercase, so offsets stay valid.
    text = "İstanbul'a gitti."
    assert len("İ".lower()) == 2
    assert char_mask(len(text), [{"start": 0, "end": 10}]).sum() == 10


def test_token_probs_to_char_probs_takes_max_and_skips_none():
    probs = token_probs_to_char_probs(6, [None, (0, 3), (2, 5)], [0.9, 0.2, 0.7])
    np.testing.assert_allclose(probs, [0.2, 0.2, 0.7, 0.7, 0.7, 0.0])


def test_encode_example_with_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("boun-tabilab/TabiBERT")
    except Exception as exc:  # offline or hub unavailable
        pytest.skip(f"tokenizer unavailable: {exc}")
    from hdtr.labels import encode_example

    enc = encode_example(tok, "Başvurular 15 Ocak'ta kapanır. Son tarih ne?", ANSWER, SPAN, max_length=64)
    flagged = [o for o, l in zip(enc["answer_offsets"], enc["labels"]) if l == 1]
    covered = "".join(ANSWER[s:e] for s, e in flagged)
    assert "25" in covered and "Ocak" in covered
    assert not enc["truncated"]
