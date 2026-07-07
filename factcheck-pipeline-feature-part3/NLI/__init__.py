"""Natural Language Inference (Feature 3).

Trains a cross-encoder classifier that maps a (premise, hypothesis)
pair to one of {entailment=0, neutral=1, contradiction=2}.

Public entry points:
    data.load_nli(...)          -- unified MNLI/ANLI/SNLI/FEVER-as-NLI loader
    baseline.train_baseline(...) -- TF-IDF + LogReg baseline
    train.train(...)             -- fine-tune the cross-encoder
    evaluate.evaluate(...)       -- per-class accuracy / macro-F1 / confusion matrix
    inference.NLIPredictor       -- pipeline-ready predict(claim, evidence)
"""

LABEL_ID2NAME = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAME2ID = {v: k for k, v in LABEL_ID2NAME.items()}
NUM_LABELS = 3
