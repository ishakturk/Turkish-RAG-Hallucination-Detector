"""Run a token-classification hallucination detector and return per-character probabilities."""

from __future__ import annotations

import time

import numpy as np

from .labels import encode_example, token_probs_to_char_probs


def load_detector(model_id: str, dtype=None, device: str | None = None, **kwargs):
    """Load tokenizer + token-classification model (pass trust_remote_code=True where needed)."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForTokenClassification.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    return tokenizer, model.to(device).eval()


def predict_char_probs(model, tokenizer, prompt: str, answer: str,
                       max_length: int = 8192) -> tuple[np.ndarray, bool]:
    """Probability that each answer character is hallucinated, and whether the prompt was cut."""
    import torch

    enc = encode_example(tokenizer, prompt, answer, (), max_length)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([enc["input_ids"]], device=device),
            attention_mask=torch.tensor([enc["attention_mask"]], device=device),
        ).logits[0]
    token_probs = logits.float().softmax(-1)[:, 1].cpu().numpy()
    return token_probs_to_char_probs(len(answer), enc["answer_offsets"], token_probs), enc["truncated"]


def predict_many(model, tokenizer, rows, max_length: int = 8192) -> tuple[list[np.ndarray], dict]:
    """Predict for an iterable of (prompt, answer) pairs; returns probabilities and timing stats."""
    probs, truncated, start = [], 0, time.perf_counter()
    for prompt, answer in rows:
        p, cut = predict_char_probs(model, tokenizer, prompt, answer, max_length)
        probs.append(p)
        truncated += cut
    elapsed = time.perf_counter() - start
    return probs, {"n": len(probs), "seconds": elapsed,
                   "sec_per_example": elapsed / max(len(probs), 1), "truncated": truncated}
