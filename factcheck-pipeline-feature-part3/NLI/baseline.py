"""TF-IDF + Logistic Regression baseline.

Purpose: give us a floor to beat and validate the data / evaluation
pipeline before the transformer training loop is trusted. The proposal
explicitly asks for "documented data showing that our trained models
perform significantly better than basic keyword matching or simple
threshold rules" -- this is that basic model.

Features: word 1-2 grams on the concatenation `premise [SEP] hypothesis`
plus a small overlap-word count feature. Classifier: multinomial LogReg.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from datasets import Dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from NLI import LABEL_ID2NAME
from NLI.data import load_nli


def _pair_text(premise: str, hypothesis: str) -> str:
    return f"{premise} [SEP] {hypothesis}"


def _overlap_features(premises: Iterable[str], hypotheses: Iterable[str]) -> csr_matrix:
    """Two hand-crafted features per row: hypothesis-token overlap ratio and length delta."""
    rows = []
    for p, h in zip(premises, hypotheses):
        p_tokens = set(p.lower().split())
        h_tokens = h.lower().split()
        overlap = sum(1 for t in h_tokens if t in p_tokens)
        overlap_ratio = overlap / max(len(h_tokens), 1)
        len_delta = len(h_tokens) - len(p.split())
        rows.append([overlap_ratio, len_delta])
    return csr_matrix(np.asarray(rows, dtype=np.float32))


@dataclass
class BaselineArtifacts:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    def predict(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform([_pair_text(p, h) for p, h in zip(premises, hypotheses)])
        extra = _overlap_features(premises, hypotheses)
        X = hstack([tfidf, extra]).tocsr()
        return self.classifier.predict(X)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineArtifacts":
        with open(path, "rb") as f:
            return pickle.load(f)


def _fit_vectorizer(texts: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=200_000,
        sublinear_tf=True,
    )
    vec.fit(texts)
    return vec


def _build_features(vec: TfidfVectorizer, ds: Dataset) -> csr_matrix:
    tfidf = vec.transform([_pair_text(p, h) for p, h in zip(ds["premise"], ds["hypothesis"])])
    extra = _overlap_features(ds["premise"], ds["hypothesis"])
    return hstack([tfidf, extra]).tocsr()


def train_baseline(
    source: str = "mnli",
    max_train_examples: int | None = 100_000,
    save_path: str | Path | None = None,
) -> BaselineArtifacts:
    """Fit the baseline and print an evaluation report on the dev split."""
    train, dev, _ = load_nli(source=source, max_train_examples=max_train_examples)

    if max_train_examples and len(train) > max_train_examples:
        train = train.shuffle(seed=42).select(range(max_train_examples))

    print(f"[baseline] fitting on {len(train)} examples from {source}")
    vec = _fit_vectorizer([_pair_text(p, h) for p, h in zip(train["premise"], train["hypothesis"])])

    X_train = _build_features(vec, train)
    y_train = np.asarray(train["label"])
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0)
    clf.fit(X_train, y_train)

    X_dev = _build_features(vec, dev)
    y_dev = np.asarray(dev["label"])
    y_pred = clf.predict(X_dev)

    print(f"[baseline] dev report ({source}):")
    print(classification_report(
        y_dev, y_pred,
        target_names=[LABEL_ID2NAME[i] for i in range(3)],
        digits=4,
    ))

    artifacts = BaselineArtifacts(vectorizer=vec, classifier=clf)
    if save_path:
        artifacts.save(save_path)
        print(f"[baseline] saved -> {save_path}")
    return artifacts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mnli",
                    choices=["mnli", "anli", "snli", "fever", "mnli+anli"])
    ap.add_argument("--max-train", type=int, default=100_000)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    train_baseline(source=args.source, max_train_examples=args.max_train, save_path=args.save)
