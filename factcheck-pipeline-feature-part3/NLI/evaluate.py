"""Evaluate a saved NLI checkpoint across MNLI, ANLI, and FEVER-as-NLI.

Produces the per-class accuracy and macro-F1 numbers required by the
proposal ("per-class accuracy across the three standard MNLI and ANLI
categories") plus a confusion matrix per split.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


def _predict(model: NLIModel, ds: Dataset, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (predicted labels, softmax probs, mean cross-entropy loss)."""
    device = next(model.model.parameters()).device
    model.model.eval()
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    total_loss, total_n = 0.0, 0

    def _collate(batch):
        premises = [b["premise"] for b in batch]
        hypotheses = [b["hypothesis"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = model.encode(premises, hypotheses)
        return enc, labels

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate)
    with torch.no_grad():
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            logits = model.model(**enc).logits
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_n += labels.size(0)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    mean_loss = total_loss / max(total_n, 1)
    return np.asarray(all_preds), np.concatenate(all_probs, axis=0), mean_loss


def _report(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, loss: float,
            runtime_seconds: float) -> Dict:
    target_names = [LABEL_ID2NAME[i] for i in range(3)]
    per_class = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    # avg softmax prob on the predicted class -- quick calibration check
    mean_confidence = float(np.mean(y_probs[np.arange(len(y_pred)), y_pred]))

    print(f"\n== {name} ==")
    print(f"n              : {len(y_true)}")
    print(f"accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"macro-F1       : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"eval_loss      : {loss:.4f}")
    print(f"mean_confidence: {mean_confidence:.4f}")
    print(f"runtime_seconds: {runtime_seconds:.1f}")
    for lbl in target_names:
        p = per_class[lbl]
        print(f"  {lbl:<14s} P={p['precision']:.4f} R={p['recall']:.4f} F1={p['f1-score']:.4f}")
    print(f"confusion matrix (rows=true, cols=pred, order=[E,N,C]):")
    for row in cm:
        print(" ", row)

    return {
        "split": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "eval_loss": loss,
        "mean_confidence": mean_confidence,
        "runtime_seconds": round(runtime_seconds, 1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _run_metadata(model_dir: str, device: str) -> Dict:
    meta = {
        "model_dir": str(model_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        meta["base_model_type"] = cfg.get("model_type")
        meta["num_hidden_layers"] = cfg.get("num_hidden_layers")
        meta["hidden_size"] = cfg.get("hidden_size")
    return meta


def evaluate(
    model_dir: str,
    batch_size: int = 64,
    include_fever: bool = True,
    device: str | None = None,
) -> Dict[str, Dict]:
    """Score the checkpoint on every held-out split and return a metrics dict."""
    eval_run_start = time.time()
    model = load_finetuned(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    results: Dict[str, Dict] = {"_meta": _run_metadata(model_dir, device)}

    mnli = load_mnli()
    for split_name in ("validation_matched", "validation_mismatched"):
        ds = mnli[split_name]
        y_true = np.asarray(ds["label"])
        t0 = time.time()
        y_pred, y_probs, loss = _predict(model, ds, batch_size=batch_size)
        results[f"mnli/{split_name}"] = _report(f"mnli/{split_name}", y_true, y_pred, y_probs, loss, time.time() - t0)

    for r in (1, 2, 3):
        anli = load_anli(r)
        ds = anli["test"]
        y_true = np.asarray(ds["label"])
        t0 = time.time()
        y_pred, y_probs, loss = _predict(model, ds, batch_size=batch_size)
        results[f"anli/r{r}/test"] = _report(f"anli/r{r}/test", y_true, y_pred, y_probs, loss, time.time() - t0)

    if include_fever:
        try:
            fever = load_fever_as_nli()
            for split_name in ("dev", "test"):
                ds = fever[split_name]
                y_true = np.asarray(ds["label"])
                t0 = time.time()
                y_pred, y_probs, loss = _predict(model, ds, batch_size=batch_size)
                results[f"fever/{split_name}"] = _report(
                    f"fever/{split_name}", y_true, y_pred, y_probs, loss, time.time() - t0)
        except Exception as e:
            print(f"[evaluate] skipping FEVER-as-NLI: {e}")
            results["fever/dev"] = {"split": "fever/dev", "error": str(e)}

    results["_meta"]["total_eval_runtime_seconds"] = round(time.time() - eval_run_start, 1)
    print(f"\n[evaluate] total eval runtime: {results['_meta']['total_eval_runtime_seconds']:.1f}s")
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
