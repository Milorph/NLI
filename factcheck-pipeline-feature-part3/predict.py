"""My entry point for the pipeline, matching the main-script interface
from the progress report ("Main Script Suggestion"):

    relationships = func_part3(input, evidence)
    supports = [r for r in relationships if r.relation == 'SUPPORTS']

input: the claim string. evidence: list of passage strings from func_part2.
Each Relationship has .relation (SUPPORTS / NOT ENOUGH INFO / REFUTES),
.evidence, and .score (confidence on the predicted class).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from NLI.inference import NLIPredictor, score_claim_evidence
from common import resource_manager

# local fallback if resource/nli_model.p isn't there yet -- see
# NLI/save_final_model.py. the merged pipeline should hit resource_manager.
_LOCAL_CHECKPOINT_DIR = "checkpoints/nli_deberta_fever/best"
_RESOURCE_FILENAME = "nli_model.p"

_LABEL_TO_RELATION = {
    "entailment": "SUPPORTS",
    "neutral": "NOT ENOUGH INFO",
    "contradiction": "REFUTES",
}

_predictor: Optional[NLIPredictor] = None


def _get_predictor() -> NLIPredictor:
    global _predictor
    if _predictor is not None:
        return _predictor

    loaded = resource_manager.load_resource(_RESOURCE_FILENAME)
    if isinstance(loaded, dict) and "hub_repo_id" in loaded:
        # deberta's too big to pickle into git, so this points at a HF
        # Hub repo instead -- see NLI/push_to_hub.py
        _predictor = NLIPredictor(loaded["hub_repo_id"])
    elif loaded is not None:
        _predictor = NLIPredictor.from_nli_model(loaded)
    else:
        _predictor = NLIPredictor(_LOCAL_CHECKPOINT_DIR)
    return _predictor


@dataclass
class Relationship:
    evidence: str
    relation: str
    score: float


def func_part3(input: str, evidence: Sequence[str]) -> List[Relationship]:
    """Score a claim against each retrieved evidence passage."""
    if not evidence:
        return []
    predictor = _get_predictor()
    rows = score_claim_evidence(predictor, claim=input, evidence=evidence)
    return [
        Relationship(
            evidence=row["evidence"],
            relation=_LABEL_TO_RELATION[row["label"]],
            score=row["probs"][row["label"]],
        )
        for row in rows
    ]


if __name__ == "__main__":
    demo = func_part3(
        "The Eiffel Tower is in Berlin.",
        ["The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
         "Mars in Paris, France."],
    )
    for r in demo:
        print(r)
