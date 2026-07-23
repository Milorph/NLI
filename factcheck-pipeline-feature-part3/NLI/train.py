"""Fine-tune the cross-encoder NLI model.

Recipe: MNLI train (optionally + all three ANLI rounds) -> HuggingFace
Trainer with weight-decay AdamW, linear warmup, cosine decay, mixed
precision when CUDA is present. Best checkpoint is picked on
MNLI-matched macro-F1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    force_fp32: bool = False  # fallback if training collapses -- deberta
    # can go unstable under bf16/fp16 without ever crashing, it just
    # converges to a useless fixed point. set True to turn off mixed precision.
    adam_epsilon: float = 1e-8  # bump to 1e-6 if grad_norm turns into nan,
    # ran into this on deberta with the default epsilon
    max_grad_norm: float = 1.0


def train(cfg: TrainConfig) -> str:
    """Run fine-tuning and return the path to the best checkpoint."""
    run_start = time.time()
    run_start_iso = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_load_start = time.time()
    train_ds, dev_ds, extra_dev = load_nli(
        source=cfg.source,
        max_train_examples=cfg.max_train_examples,
        seed=cfg.seed,
    )
    data_load_seconds = time.time() - data_load_start

    nli = build_model(cfg.model_name, max_length=cfg.max_length)
    train_tok = _tokenize_dataset(train_ds, nli)
    dev_tok = _tokenize_dataset(dev_ds, nli)

    # warmup_ratio kept rounding down to 0 steps on my transformers
    # version, so I just compute warmup_steps myself
    steps_per_epoch = math.ceil(len(train_tok) / cfg.train_batch_size)
    total_steps = int(steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not cfg.force_fp32,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported() and not cfg.force_fp32,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=nli.model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        processing_class=nli.tokenizer,
        data_collator=DataCollatorWithPadding(nli.tokenizer),
        compute_metrics=_compute_metrics,
    )

    fit_start = time.time()
    trainer.train()
    fit_seconds = time.time() - fit_start

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

    total_seconds = time.time() - run_start
    run_info = {
        "source": cfg.source,
        "model_name": cfg.model_name,
        "epochs": cfg.epochs,
        "train_batch_size": cfg.train_batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": warmup_steps,
        "seed": cfg.seed,
        "force_fp32": cfg.force_fp32,
        "adam_epsilon": cfg.adam_epsilon,
        "max_grad_norm": cfg.max_grad_norm,
        "train_examples": len(train_tok),
        "dev_examples": len(dev_tok),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": run_start_iso,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "data_load_seconds": round(data_load_seconds, 1),
        "fit_seconds": round(fit_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "fit_hms": time.strftime("%H:%M:%S", time.gmtime(fit_seconds)),
        "total_hms": time.strftime("%H:%M:%S", time.gmtime(total_seconds)),
    }
    with open(os.path.join(final_dir, "run_timing.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[train] total run time: {run_info['total_hms']} (fit: {run_info['fit_hms']}) -> "
          f"{os.path.join(final_dir, 'run_timing.json')}")

    return final_dir


def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", default=DEFAULT_BACKBONE)
    ap.add_argument("--source", default="mnli",
                    choices=["mnli", "anli", "snli", "fever", "mnli+anli", "mnli+anli+fever"])
    ap.add_argument("--output-dir", default="checkpoints/nli")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", dest="learning_rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train", dest="max_train_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", dest="force_fp32", action="store_true",
                    help="Disable bf16/fp16 mixed precision (stability fallback for architectures "
                         "like DeBERTa-v3 that can collapse mid-training under reduced precision)")
    ap.add_argument("--adam-epsilon", type=float, default=1e-8,
                    help="AdamW epsilon. Raise to 1e-6 as a stability fix for NaN gradients "
                         "(seen on DeBERTa-v3 with the default 1e-8).")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(_parse_args())
