"""Cross-encoder NLI model wrapper.

Following Reimers & Gurevych (Sentence-BERT, 2019), a *cross-encoder* is
the right architecture when we want a single {entail, neutral, contradict}
label for a specific (premise, hypothesis) pair. The two sentences are
fed jointly through BERT with a [SEP] token; the [CLS] hidden state is
projected to 3 logits.

We deliberately use HuggingFace's ``AutoModelForSequenceClassification``
so the checkpoint transparently loads either:
    * a raw BERT / RoBERTa backbone (train from a general MLM), or
    * a pretrained NLI cross-encoder such as
      ``cross-encoder/nli-deberta-v3-base`` (fine-tune on our data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from NLI import LABEL_ID2NAME, LABEL_NAME2ID, NUM_LABELS


DEFAULT_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
"""Small, fast default. For higher accuracy swap in bert-base-uncased or
``cross-encoder/nli-deberta-v3-base`` via the ``model_name`` argument."""


@dataclass
class NLIModel:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    max_length: int = 128

    def encode(self, premises: List[str], hypotheses: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_logits(self, premises: List[str], hypotheses: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in self.encode(premises, hypotheses).items()}
        self.model.eval()
        return self.model(**batch).logits.detach().cpu()


def build_model(
    model_name: str = DEFAULT_BACKBONE,
    num_labels: int = NUM_LABELS,
    max_length: int = 128,
) -> NLIModel:
    """Load tokenizer + model with a 3-way classification head."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=LABEL_ID2NAME,
        label2id=LABEL_NAME2ID,
        ignore_mismatched_sizes=True,
    )
    return NLIModel(tokenizer=tokenizer, model=model, max_length=max_length)


def load_finetuned(model_dir: Union[str, "os.PathLike"], max_length: int = 128) -> NLIModel:
    """Reload a checkpoint saved by ``Trainer.save_model`` / ``model.save_pretrained``."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return NLIModel(tokenizer=tokenizer, model=model, max_length=max_length)
