"""Fine-tuning a token classifier for hallucination detection (Hugging Face Trainer).

"Our recipe" (docs/technical-roadmap.md §2.3): class-weighted or focal loss, warmup + linear decay,
evaluation during training, best-checkpoint selection, length-grouped batches, bf16.
The "Newmind-style" recipe (R0) is the same code with plain CE, fixed epochs and no evaluation.
"""

from __future__ import annotations

import json
import os
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .labels import IGNORE_INDEX, encode_example, token_probs_to_char_probs


@dataclass
class RunConfig:
    name: str
    model_id: str
    lr: float = 2e-5
    epochs: float = 1.0
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 4096
    loss: str = "weighted_ce"          # "ce" | "weighted_ce" | "focal"
    focal_gamma: float = 2.0
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    split_words: bool = False
    seed: int = 42
    evals_per_epoch: int = 0           # 0 = no evaluation during training
    early_stopping_patience: int = 0   # 0 = off
    select_metric: str = "word_f1"     # checkpoint choice on validation: word_f1 | span_f1 | token_f1
    gradient_checkpointing: bool = True
    save_checkpoints: bool = False
    trust_remote_code: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        import yaml

        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def encode_frame(tokenizer, frame, max_length: int, split_words: bool = False) -> list[dict]:
    """Encode every row of a frame with ``prompt``, ``answer`` and ``labels`` columns."""
    return [encode_example(tokenizer, prompt, answer, spans, max_length, split_words=split_words)
            for prompt, answer, spans in zip(frame["prompt"], frame["answer"], frame["labels"])]


def build_dataset(tokenizer, frame, max_length: int, split_words: bool = False, encodings=None):
    """HF Dataset of input ids, attention mask and labels (pass ``encodings`` to reuse them)."""
    from datasets import Dataset

    encodings = encodings if encodings is not None else encode_frame(tokenizer, frame, max_length, split_words)
    rows = {k: [enc[k] for enc in encodings] for k in ("input_ids", "attention_mask", "labels")}
    ds = Dataset.from_dict(rows)
    ds.info.description = json.dumps({"truncated": int(sum(enc["truncated"] for enc in encodings))})
    return ds


def class_weights(dataset) -> list[float]:
    """[1, sqrt(neg/pos)] over labeled tokens: up-weights the rare hallucinated class, but gently."""
    pos = neg = 0
    for labels in dataset["labels"]:
        arr = np.asarray(labels)
        pos += int((arr == 1).sum())
        neg += int((arr == 0).sum())
    return [1.0, math.sqrt(neg / max(pos, 1))]


def token_loss(logits, labels, kind: str = "weighted_ce", weights=None, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F

    logits = logits.view(-1, logits.size(-1)).float()
    labels = labels.view(-1)
    if kind == "focal":
        ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="none")
        mask = labels != IGNORE_INDEX
        pt = torch.exp(-ce[mask])
        return ((1 - pt) ** gamma * ce[mask]).mean()
    w = torch.tensor(weights, device=logits.device) if kind == "weighted_ce" and weights else None
    return F.cross_entropy(logits, labels, weight=w, ignore_index=IGNORE_INDEX)


def _trainer_class():
    from transformers import Trainer

    class HallucinationTrainer(Trainer):
        def __init__(self, *args, loss_kind="weighted_ce", loss_weights=None, focal_gamma=2.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_kind, self.loss_weights, self.focal_gamma = loss_kind, loss_weights, focal_gamma

        def _get_eval_sampler(self, eval_dataset):
            # group_by_length would also shuffle evaluation by length; word metrics need the
            # predictions in example order to map tokens back to each answer.
            from torch.utils.data import SequentialSampler

            return SequentialSampler(eval_dataset) if eval_dataset is not None else None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = token_loss(outputs.logits, labels, self.loss_kind, self.loss_weights, self.focal_gamma)
            return (loss, outputs) if return_outputs else loss

    return HallucinationTrainer


def _preprocess_logits(logits, labels):
    """Keep only P(hallucinated) per token to save memory during evaluation."""
    return logits.float().softmax(-1)[..., 1]


def token_metrics(eval_pred) -> dict:
    """Token-level F1 (hallucinated class) and AUROC; used only to pick checkpoints within a run."""
    from sklearn.metrics import roc_auc_score

    probs, labels = eval_pred
    mask = labels != IGNORE_INDEX
    y, p = labels[mask], probs[mask]
    pred = p >= 0.5
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auroc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")
    return {"token_f1": f1, "token_precision": prec, "token_recall": rec, "token_auroc": auroc}


def word_metrics_fn(frame, encodings, threshold: float = 0.5):
    """``compute_metrics`` for the Trainer: token metrics plus our word, span and example
    metrics on the validation answers (predictions mapped back to characters, then words)."""
    from .metrics import evaluate
    from .postprocess import char_probs_to_spans

    answers = list(frame["answer"])
    golds = [list(x) for x in frame["labels"]]
    tasks = list(frame["task_type"]) if "task_type" in frame else [None] * len(answers)

    def compute(eval_pred) -> dict:
        probs, labels = eval_pred
        out = token_metrics((probs, labels))
        records = []
        for i, enc in enumerate(encodings):
            n = len(enc["input_ids"])
            char_probs = token_probs_to_char_probs(len(answers[i]), enc["answer_offsets"], probs[i, :n])
            records.append({"answer": answers[i], "gold_spans": golds[i], "task_type": tasks[i],
                            "pred_spans": char_probs_to_spans(answers[i], char_probs, threshold),
                            "char_probs": char_probs})
        m = evaluate(records, by_task=False)["overall"]
        out.update(word_f1=m["word"]["f1"], span_f1=m["span"]["f1"], example_f1=m["example"]["f1"],
                   word_auroc=m.get("word_auroc", float("nan")))
        return out

    return compute


def train(cfg: RunConfig, train_frame, val_frame, output_dir: str | Path) -> dict:
    """Train one run; returns the trainer, tokenizer, model and a history/summary dict."""
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, EarlyStoppingCallback,
                              TrainingArguments, set_seed)

    set_seed(cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kw = {"trust_remote_code": True} if cfg.trust_remote_code else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kw)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_id, num_labels=2, id2label={0: "supported", 1: "hallucinated"},
        label2id={"supported": 0, "hallucinated": 1}, ignore_mismatched_sizes=True, **kw)

    t0 = time.perf_counter()
    train_ds = build_dataset(tokenizer, train_frame, cfg.max_length, cfg.split_words)
    val_enc = encode_frame(tokenizer, val_frame, cfg.max_length, cfg.split_words) if val_frame is not None else None
    val_ds = build_dataset(tokenizer, val_frame, cfg.max_length, encodings=val_enc) if val_enc is not None else None
    encode_sec = time.perf_counter() - t0
    weights = class_weights(train_ds) if cfg.loss == "weighted_ce" else None

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    evaluate = cfg.evals_per_epoch > 0 and val_ds is not None
    eval_steps = max(1, steps_per_epoch // cfg.evals_per_epoch) if evaluate else None
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        group_by_length=True,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=eval_steps,
        save_strategy=("steps" if evaluate else "epoch") if cfg.save_checkpoints else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=evaluate and cfg.save_checkpoints,
        metric_for_best_model=cfg.select_metric,
        greater_is_better=True,
        logging_steps=max(1, steps_per_epoch // 50),
        report_to="none",
        seed=cfg.seed,
        # Worker processes hang on Windows (spawn start method); Colab is Linux.
        dataloader_num_workers=0 if os.name == "nt" else 2,
    )
    callbacks = []
    if evaluate and cfg.early_stopping_patience and cfg.save_checkpoints:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))
    trainer = _trainer_class()(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds if evaluate else None,
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
        compute_metrics=word_metrics_fn(val_frame, val_enc) if evaluate else None,
        preprocess_logits_for_metrics=_preprocess_logits if evaluate else None,
        callbacks=callbacks, loss_kind=cfg.loss, loss_weights=weights, focal_gamma=cfg.focal_gamma,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    train_sec = time.perf_counter() - t0
    summary = {
        "config": asdict(cfg),
        "train_examples": len(train_ds),
        "train_tokens": int(sum(len(x) for x in train_ds["input_ids"])),
        "truncated_train": json.loads(train_ds.info.description)["truncated"],
        "class_weights": weights,
        "encode_sec": round(encode_sec, 1),
        "train_sec": round(train_sec, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    return {"trainer": trainer, "tokenizer": tokenizer, "model": trainer.model, "summary": summary}
