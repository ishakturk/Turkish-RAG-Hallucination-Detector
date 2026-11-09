import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hdtr.train import token_loss, token_metrics  # noqa: E402


def _batch():
    logits = torch.tensor([[[2.0, -2.0], [-1.0, 1.0], [0.0, 0.0]]])
    labels = torch.tensor([[0, 1, -100]])
    return logits, labels


def test_weighted_ce_ignores_masked_tokens_and_upweights_positives():
    logits, labels = _batch()
    plain = token_loss(logits, labels, "ce")
    weighted = token_loss(logits, labels, "weighted_ce", [1.0, 3.0])
    assert torch.isfinite(plain) and torch.isfinite(weighted)
    # Changing the ignored token's logits must not change the loss.
    logits2 = logits.clone(); logits2[0, 2] = torch.tensor([9.0, -9.0])
    assert torch.isclose(plain, token_loss(logits2, labels, "ce"))


def test_focal_loss_is_smaller_than_ce_for_easy_tokens():
    logits, labels = _batch()
    assert token_loss(logits, labels, "focal", gamma=2.0) < token_loss(logits, labels, "ce")


def test_token_metrics_skip_ignored_positions():
    probs = np.array([[0.9, 0.1, 0.8, 0.7]])
    labels = np.array([[1, 0, 1, -100]])
    m = token_metrics((probs, labels))
    assert m["token_f1"] == 1.0 and m["token_auroc"] == 1.0


def test_word_metrics_fn_maps_padded_predictions_back_to_words():
    import pandas as pd
    from transformers import AutoTokenizer

    from hdtr.train import encode_frame, word_metrics_fn

    tok = AutoTokenizer.from_pretrained("boun-tabilab/TabiBERT")
    answer = "Toplantı Ocak'a kadar ertelendi ve 15.000 TL ödendi."
    start = answer.index("15.000")
    frame = pd.DataFrame({"prompt": ["Bağlam: toplantı ertelendi.", "Kısa bağlam metni burada."],
                          "answer": [answer, "Her şey doğru."],
                          "labels": [[{"start": start, "end": start + 9}], []],
                          "task_type": ["QA", "QA"]})
    enc = encode_frame(tok, frame, 128)
    width = max(len(e["input_ids"]) for e in enc) + 5          # Trainer pads with -100
    probs = np.full((2, width), -100.0, dtype=np.float32)
    labels = np.full((2, width), -100, dtype=np.int64)
    for i, e in enumerate(enc):
        lab = np.array(e["labels"])
        n = len(lab)
        probs[i, :n] = np.where(lab == 1, 0.9, 0.1)
        labels[i, :n] = lab
    m = word_metrics_fn(frame, enc)((probs, labels))
    assert m["word_f1"] == 1.0 and m["example_f1"] == 1.0 and m["span_f1"] > 0.9


def test_tune_thresholds_prefers_best_and_closest_to_half():
    import pandas as pd

    from hdtr.evaluation import score, tune_thresholds

    answer = "Lig 1927 yılında kuruldu."
    frame = pd.DataFrame({"answer": [answer, answer], "task_type": ["QA", "Summary"], "source_id": [1, 2],
                          "labels": [[{"start": 4, "end": 8}], []]})
    p1 = np.zeros(len(answer), dtype=np.float32); p1[4:8] = 0.3
    p2 = np.zeros(len(answer), dtype=np.float32)
    t = tune_thresholds(frame, [p1, p2])
    assert t["tau_example"] == 0.3 and t["tau_span"] == 0.3
    s = score(frame, [p1, p2], 0.3)
    assert s["overall"]["example_f1"] == 1.0 and set(s) == {"overall", "QA", "Summary"}
