"""Pipeline-ready inference for the NLI stage.

Feature 3's job in the fact-checking pipeline is to take:
    * a claim (from Feature 1 - Claim Extraction), and
    * one or more evidence passages (from Feature 2 - Evidence Retrieval)
and produce, for each claim-evidence pair, a probability distribution
over {entailment, neutral, contradiction}. The aggregation stage
(Feature 4, Kusuma) consumes those distributions to make the final
SUPPORTS / NOT ENOUGH INFO / REFUTES call.

Consistent with training, the evidence text is treated as the premise
and the claim as the hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from NLI import LABEL_ID2NAME
from NLI.model import NLIModel, load_finetuned


@dataclass
class NLIPrediction:
    label: str
    label_id: int
    probs: dict  # {"entailment": float, "neutral": float, "contradiction": float}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "probs": self.probs,
        }


class NLIPredictor:
    """Thin batching wrapper. One instance per process — reuse it."""

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 32,
    ):
        self.model: NLIModel = load_finetuned(model_dir, max_length=max_length)
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.model.to(device)
        self.model.model.eval()

    def _softmax(self, logits: torch.Tensor) -> np.ndarray:
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
    ) -> List[NLIPrediction]:
        """Batched inference on (premise, hypothesis) pairs."""
        results: List[NLIPrediction] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]
            logits = self.model.predict_logits(premises, hypotheses)
            probs = self._softmax(logits)
            for row in probs:
                label_id = int(np.argmax(row))
                results.append(NLIPrediction(
                    label=LABEL_ID2NAME[label_id],
                    label_id=label_id,
                    probs={LABEL_ID2NAME[i]: float(row[i]) for i in range(3)},
                ))
        return results

    def predict(
        self,
        claim: str,
        evidence: str | Iterable[str],
    ) -> List[NLIPrediction]:
        """Score a single claim against one or many evidence strings.

        Evidence goes in as the premise, claim as the hypothesis --
        matching the training convention.
        """
        if isinstance(evidence, str):
            evidence_list: List[str] = [evidence]
        else:
            evidence_list = list(evidence)
        pairs = [(ev, claim) for ev in evidence_list]
        return self.predict_pairs(pairs)


def score_claim_evidence(
    predictor: NLIPredictor,
    claim: str,
    evidence: Sequence[str],
) -> List[dict]:
    """Convenience: returns a list of dicts ready to hand to the aggregator."""
    preds = predictor.predict(claim, evidence)
    return [
        {"evidence": ev, **pred.to_dict()}
        for ev, pred in zip(evidence, preds)
    ]
