"""Evaluate a saved NLI checkpoint across MNLI, ANLI, and FEVER-as-NLI.

Produces the per-class accuracy and macro-F1 numbers required by the
proposal ("per-class accuracy across the three standard MNLI and ANLI
categories") plus a confusion matrix per split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from NLI import LABEL_ID2NAME
from NLI.data import (
    load_anli,
    load_fever_as_nli,
    load_mnli,
)
from NLI.model import NLIModel, load_finetuned


def _predict(model: NLIModel, ds: Dataset, batch_size: int = 64) -> np.ndarray:
    device = next(model.model.parameters()).device
    model.model.eval()
    all_preds: List[int] = []

    def _collate(batch):
        premises = [b["premise"] for b in batch]
        hypotheses = [b["hypothesis"] for b in batch]
        return model.encode(premises, hypotheses)

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model.model(**batch).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return np.asarray(all_preds)


def _report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    target_names = [LABEL_ID2NAME[i] for i in range(3)]
    per_class = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    print(f"\n== {name} ==")
    print(f"accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"macro-F1       : {f1_score(y_true, y_pred, average='macro'):.4f}")
    for lbl in target_names:
        p = per_class[lbl]
        print(f"  {lbl:<14s} P={p['precision']:.4f} R={p['recall']:.4f} F1={p['f1-score']:.4f}")
    print(f"confusion matrix (rows=true, cols=pred, order=[E,N,C]):")
    for row in cm:
        print(" ", row)

    return {
        "split": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate(
    model_dir: str,
    batch_size: int = 64,
    include_fever: bool = True,
    device: str | None = None,
) -> Dict[str, Dict]:
    """Score the checkpoint on every held-out split and return a metrics dict."""
    model = load_finetuned(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    results: Dict[str, Dict] = {}

    mnli = load_mnli()
    for split_name in ("validation_matched", "validation_mismatched"):
        ds = mnli[split_name]
        y_true = np.asarray(ds["label"])
        y_pred = _predict(model, ds, batch_size=batch_size)
        results[f"mnli/{split_name}"] = _report(f"mnli/{split_name}", y_true, y_pred)

    for r in (1, 2, 3):
        anli = load_anli(r)
        ds = anli["test"]
        y_true = np.asarray(ds["label"])
        y_pred = _predict(model, ds, batch_size=batch_size)
        results[f"anli/r{r}/test"] = _report(f"anli/r{r}/test", y_true, y_pred)

    if include_fever:
        try:
            fever = load_fever_as_nli()
            ds = fever["dev"] if "dev" in fever else fever["test"]
            y_true = np.asarray(ds["label"])
            y_pred = _predict(model, ds, batch_size=batch_size)
            results["fever/dev"] = _report("fever/dev", y_true, y_pred)
        except Exception as e:
            print(f"[evaluate] skipping FEVER-as-NLI: {e}")

    return results


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Path to a saved cross-encoder checkpoint")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-fever", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to dump the metrics dict as JSON")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    metrics = evaluate(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        include_fever=not args.no_fever,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"[evaluate] metrics -> {args.out}")
