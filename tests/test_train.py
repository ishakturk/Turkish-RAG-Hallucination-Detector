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
