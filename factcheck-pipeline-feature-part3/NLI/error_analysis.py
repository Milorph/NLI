"""Error analysis for a fine-tuned NLI checkpoint.

evaluate.py only gives me aggregate accuracy/macro-F1. This digs deeper --
per-genre accuracy (MNLI has a genre column I normally drop) and actual
misclassified examples for each error type, so I can see what kind of
mistakes the model is making, not just the numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from NLI import LABEL_ID2NAME
from NLI.model import NLIModel, load_finetuned
from common import dataset as common_dataset


def _predict_all(model: NLIModel, premises: List[str], hypotheses: List[str], batch_size: int = 64):
    device = next(model.model.parameters()).device
    model.model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(premises), batch_size):
            p = premises[start:start + batch_size]
            h = hypotheses[start:start + batch_size]
            batch = {k: v.to(device) for k, v in model.encode(p, h).items()}
            logits = model.model(**batch).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def analyze(model_dir: str, examples_per_error: int = 3, batch_size: int = 64, out_path: str | None = None) -> Dict:
    run_start = time.time()
    model = load_finetuned(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    # Raw MNLI-matched dev, with genre kept (NLI.data._clean drops it).
    _, val_m, _ = common_dataset.find_logical_relationship()
    val_m = val_m.filter(lambda ex: ex["label"] in (0, 1, 2) and ex["premise"] and ex["hypothesis"])

    premises = val_m["premise"]
    hypotheses = val_m["hypothesis"]
    genres = val_m["genre"]
    y_true = np.asarray(val_m["label"])

    probs = _predict_all(model, premises, hypotheses, batch_size=batch_size)
    y_pred = np.argmax(probs, axis=-1)
    confidence = probs[np.arange(len(y_pred)), y_pred]

    names = [LABEL_ID2NAME[i] for i in range(3)]
    acc = (y_true == y_pred).mean()
    print(f"\n== Overall: {len(y_true)} examples, accuracy={acc:.4f} ==")

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = " " * 16 + "".join(f"{n:>14s}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:<16s}" + "".join(f"{v:>14d}" for v in row))

    # --- per-genre accuracy ---
    print("\nPer-genre accuracy:")
    genre_correct: Dict[str, int] = defaultdict(int)
    genre_total: Dict[str, int] = defaultdict(int)
    for g, t, p in zip(genres, y_true, y_pred):
        genre_total[g] += 1
        if t == p:
            genre_correct[g] += 1
    per_genre = {}
    for g in sorted(genre_total, key=lambda g: -genre_total[g]):
        n = genre_total[g]
        genre_acc = genre_correct[g] / n
        per_genre[g] = {"n": n, "accuracy": genre_acc}
        print(f"  {g:<12s} n={n:<6d} acc={genre_acc:.4f}")

    # --- example errors per error type ---
    print("\nExample errors by (true -> predicted):")
    error_examples: Dict[str, List[Dict]] = {}
    for t in range(3):
        for p in range(3):
            if t == p:
                continue
            mask = (y_true == t) & (y_pred == p)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            # sort by confidence -- most interesting mistakes are the ones
            # where the model was sure and still wrong
            idxs = idxs[np.argsort(-confidence[idxs])][:examples_per_error]
            key = f"{names[t]}->{names[p]}"
            error_examples[key] = []
            print(f"\n  {names[t]} -> {names[p]}  ({mask.sum()} total)")
            for i in idxs:
                print(f"    premise:    {premises[i]}")
                print(f"    hypothesis: {hypotheses[i]}")
                print(f"    genre: {genres[i]}  confidence: {confidence[i]:.3f}  probs: "
                      f"E={probs[i][0]:.2f} N={probs[i][1]:.2f} C={probs[i][2]:.2f}")
                print()
                error_examples[key].append({
                    "premise": premises[i],
                    "hypothesis": hypotheses[i],
                    "genre": genres[i],
                    "confidence": float(confidence[i]),
                    "probs": {"entailment": float(probs[i][0]), "neutral": float(probs[i][1]),
                              "contradiction": float(probs[i][2])},
                })

    result = {
        "model_dir": model_dir,
        "n": len(y_true),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": names,
        "per_genre_accuracy": per_genre,
        "error_examples_by_type": {k: v for k, v in error_examples.items() if v},
        "total_error_counts": {
            f"{names[t]}->{names[p]}": int(((y_true == t) & (y_pred == p)).sum())
            for t in range(3) for p in range(3) if t != p
        },
        "runtime_seconds": round(time.time() - run_start, 1),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[error_analysis] results -> {out_path}")
    return result


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Path to a saved cross-encoder checkpoint")
    ap.add_argument("--examples-per-error", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to dump the full analysis as JSON")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(args.model_dir, examples_per_error=args.examples_per_error,
            batch_size=args.batch_size, out_path=args.out)
