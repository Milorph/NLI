"""Fine-tune the cross-encoder NLI model.

Recipe: MNLI train (optionally + all three ANLI rounds) -> HuggingFace
Trainer with weight-decay AdamW, linear warmup, cosine decay, mixed
precision when CUDA is present. Best checkpoint is picked on
MNLI-matched macro-F1.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from NLI import LABEL_ID2NAME
from NLI.data import load_nli
from NLI.model import DEFAULT_BACKBONE, NLIModel, build_model


def _tokenize_dataset(ds: Dataset, model: NLIModel) -> Dataset:
    def _tok(batch):
        return model.tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=model.max_length,
        )
    ds = ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names if c not in {"label"}])
    return ds


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_entailment": f1_score(labels, preds, labels=[0], average="macro"),
        "f1_neutral":    f1_score(labels, preds, labels=[1], average="macro"),
        "f1_contradict": f1_score(labels, preds, labels=[2], average="macro"),
    }


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_BACKBONE
    source: str = "mnli"                   # or "mnli+anli"
    output_dir: str = "checkpoints/nli"
    max_length: int = 128
    epochs: float = 3.0
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_train_examples: int | None = None
    seed: int = 42


def train(cfg: TrainConfig) -> str:
    """Run fine-tuning and return the path to the best checkpoint."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds, dev_ds, extra_dev = load_nli(
        source=cfg.source,
        max_train_examples=cfg.max_train_examples,
        seed=cfg.seed,
    )

    nli = build_model(cfg.model_name, max_length=cfg.max_length)
    train_tok = _tokenize_dataset(train_ds, nli)
    dev_tok = _tokenize_dataset(dev_ds, nli)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=nli.model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=nli.tokenizer,
        data_collator=DataCollatorWithPadding(nli.tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    if extra_dev is not None:
        extra_tok = _tokenize_dataset(extra_dev, nli)
        metrics = trainer.evaluate(extra_tok)
        print("[train] extra eval split:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    final_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(final_dir)
    nli.tokenizer.save_pretrained(final_dir)
    print(f"[train] best checkpoint saved to {final_dir}")

    preds = trainer.predict(dev_tok)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(classification_report(
        preds.label_ids, y_pred,
        target_names=[LABEL_ID2NAME[i] for i in range(3)],
        digits=4,
    ))

    return final_dir


def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", default=DEFAULT_BACKBONE)
    ap.add_argument("--source", default="mnli",
                    choices=["mnli", "anli", "snli", "fever", "mnli+anli"])
    ap.add_argument("--output-dir", default="checkpoints/nli")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", dest="learning_rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train", dest="max_train_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(_parse_args())
