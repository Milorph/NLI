# Feature 3 — Natural Language Inference

Cross-encoder classifier that maps a `(premise, hypothesis)` pair to
one of:

| id | label         |
|----|---------------|
| 0  | entailment    |
| 1  | neutral       |
| 2  | contradiction |

## Layout

```
NLI/
├── __init__.py       # label maps
├── data.py           # unified MNLI / ANLI / SNLI / FEVER-as-NLI loader
├── baseline.py       # TF-IDF + Logistic Regression sanity baseline
├── model.py          # cross-encoder wrapper (Sentence-BERT-style)
├── train.py          # HF Trainer fine-tuning script
├── evaluate.py       # per-class accuracy / macro-F1 / confusion matrix
├── inference.py      # pipeline-ready NLIPredictor
├── preprocess.py     # exploratory tokenization
└── dataCheck.py      # exploratory dataset inspection
```

## Setup

```bash
pip install -e .
```

## Reproducing the deliverables

### 1. Baseline (proposal: "beat basic keyword / threshold rules")

```bash
python -m NLI.baseline --source mnli --max-train 100000 \
    --save checkpoints/nli_baseline.pkl
```

Prints per-class precision/recall/F1 on MNLI-matched.

### 2. Fine-tune cross-encoder

MNLI only:

```bash
python -m NLI.train --source mnli \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 3 --train-batch-size 32 --lr 2e-5 \
    --output-dir checkpoints/nli
```

Adversarial hardening (Adversarial NLI paper recipe):

```bash
python -m NLI.train --source mnli+anli \
    --model cross-encoder/nli-deberta-v3-base \
    --epochs 2 --output-dir checkpoints/nli_anli
```

`--max-train` caps training-set size for quick smoke tests.

### 3. Evaluate

Runs MNLI-matched, MNLI-mismatched, ANLI r1/r2/r3, and FEVER-as-NLI:

```bash
python -m NLI.evaluate checkpoints/nli/best --out results/nli_metrics.json
```

### 4. Pipeline hand-off

Feature 4 (aggregation) imports the predictor directly:

```python
from NLI.inference import NLIPredictor, score_claim_evidence

predictor = NLIPredictor("checkpoints/nli/best")
rows = score_claim_evidence(
    predictor,
    claim="The Eiffel Tower is in Berlin.",
    evidence=[
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    ],
)
# rows -> [{"evidence": "...", "label": "contradiction", "label_id": 2,
#          "probs": {"entailment": 0.02, "neutral": 0.05, "contradiction": 0.93}}]
```

Evidence is the premise, claim is the hypothesis — same convention as
training, so calibration transfers.

## Metrics reported

For every eval split we log:

- accuracy
- macro-F1
- per-class precision / recall / F1 for `entailment`, `neutral`,
  `contradiction`
- 3x3 confusion matrix (rows = true, cols = pred)

This matches the "per-class accuracy across MNLI and ANLI" target in
the proposal.

## Model choices

- **Default backbone:** `sentence-transformers/all-MiniLM-L6-v2` —
  small, cheap, runs on a laptop / free Colab.
- **Recommended for final numbers:** `cross-encoder/nli-deberta-v3-base`
  — already NLI-pretrained, fine-tunes fast, hits published-baseline
  quality on MNLI.
- Any HF `AutoModelForSequenceClassification` checkpoint works
  via `--model`.
