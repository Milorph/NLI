"""Dump a trained checkpoint into resource/ so it's git-tracked and
loadable straight from resource_manager for the merged pipeline --
checkpoints/ itself is gitignored so it never makes it to origin.

Usage:
    python -m NLI.save_final_model checkpoints/nli_minilm_fever/best
"""

from __future__ import annotations

import argparse

from NLI.model import load_finetuned
from common import resource_manager


def save_final_model(model_dir: str, filename: str = "nli_model.p") -> None:
    nli_model = load_finetuned(model_dir)
    resource_manager.save_resource(nli_model, filename)
    print(f"[save_final_model] {model_dir} -> resource/{filename}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--filename", default="nli_model.p")
    args = ap.parse_args()
    save_final_model(args.model_dir, args.filename)
