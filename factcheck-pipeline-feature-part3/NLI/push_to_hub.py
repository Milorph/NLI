"""Push a checkpoint too big for git (looking at you, deberta) to the HF
Hub, then point resource/nli_model.p at it instead of the raw weights.

Needs you logged in first:
    huggingface-cli login

Usage:
    python -m NLI.push_to_hub checkpoints/nli_deberta_fever/best my-hf-username/nli-deberta-fever
"""

from __future__ import annotations

import argparse

from NLI.model import load_finetuned
from common import resource_manager


def push_to_hub(model_dir: str, repo_id: str, filename: str = "nli_model.p") -> None:
    nli = load_finetuned(model_dir)
    print(f"[push_to_hub] uploading {model_dir} -> {repo_id} ...")
    nli.model.push_to_hub(repo_id)
    nli.tokenizer.push_to_hub(repo_id)

    resource_manager.save_resource({"hub_repo_id": repo_id}, filename)
    print(f"[push_to_hub] done. resource/{filename} now points at {repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("repo_id", help="e.g. your-hf-username/nli-deberta-fever")
    ap.add_argument("--filename", default="nli_model.p")
    args = ap.parse_args()
    push_to_hub(args.model_dir, args.repo_id, args.filename)
