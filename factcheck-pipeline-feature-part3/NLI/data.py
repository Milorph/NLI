"""Unified NLI data loading.

Normalizes every source to the same schema so downstream code doesn't
have to care where an example came from:

    {"premise": str, "hypothesis": str, "label": int}

Label convention (matches MNLI / ANLI):
    0 = entailment
    1 = neutral
    2 = contradiction

SNLI ships with -1 for "no gold consensus" examples; those are dropped.
FEVER (as used in this pipeline) is folded into NLI shape by treating
the retrieved evidence as the premise and the claim as the hypothesis:
    SUPPORTS         -> 0 entailment
    NOT ENOUGH INFO  -> 1 neutral
    REFUTES          -> 2 contradiction
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset

from common import dataset as common_dataset


_KEEP_COLUMNS = ["premise", "hypothesis", "label"]


def _clean(ds: Dataset) -> Dataset:
    """Drop rows with -1 labels or empty text, keep only the shared columns."""

    def _ok(ex):
        if ex["label"] not in (0, 1, 2):
            return False
        p, h = ex.get("premise"), ex.get("hypothesis")
        if not p or not h:
            return False
        if not p.strip() or not h.strip():
            return False
        return True

    ds = ds.filter(_ok)
    keep = [c for c in _KEEP_COLUMNS if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    # cast label to plain int64 everywhere -- mnli/anli use ClassLabel,
    # fever uses Value('int64'), and concatenate_datasets just refuses
    # to mix the two
    ds = ds.cast_column("label", Value("int64"))
    return ds


def load_mnli() -> DatasetDict:
    """MultiNLI: train / validation_matched / validation_mismatched."""
    train, val_m, val_mm = common_dataset.find_logical_relationship()
    return DatasetDict({
        "train": _clean(train),
        "validation_matched": _clean(val_m),
        "validation_mismatched": _clean(val_mm),
    })


def load_anli(round_num: Literal[1, 2, 3] = 1) -> DatasetDict:
    """Adversarial NLI round r{n}: train / dev / test."""
    tr, dv, te = common_dataset.find_logical_relationship_improve(round_num)
    return DatasetDict({
        "train": _clean(tr),
        "dev": _clean(dv),
        "test": _clean(te),
    })


def load_snli() -> DatasetDict:
    """SNLI backup. -1 labels are filtered out by _clean."""
    dt = load_dataset("stanfordnlp/snli")
    return DatasetDict({
        "train": _clean(dt["train"]),
        "validation": _clean(dt["validation"]),
        "test": _clean(dt["test"]),
    })


_FEVER_LABEL_MAP = {
    "SUPPORTS": 0,
    "NOT ENOUGH INFO": 1,
    "REFUTES": 2,
}


def _fever_gold_to_nli(row: Dict) -> Dict:
    """Cast a copenlu/fever_gold_evidence row into NLI shape.

    ``evidence`` is a list of [wiki_title, sentence_id, sentence_text]
    triples; I join their sentence texts together as the premise. Claim
    is the hypothesis.
    """
    triples = row.get("evidence") or []
    sentences = [t[2] for t in triples if t and len(t) > 2 and t[2]]
    premise = " ".join(sentences)
    hypothesis = row.get("claim") or ""
    raw = row.get("label")
    label = _FEVER_LABEL_MAP.get(str(raw).upper(), -1) if raw is not None else -1
    return {"premise": premise, "hypothesis": hypothesis, "label": label}


def load_fever_as_nli() -> DatasetDict:
    """FEVER (claim + gold evidence + verdict) cast to NLI shape.

    Loads ``copenlu/fever_gold_evidence`` directly rather than going
    through ``common.dataset.classify_evidence()`` -- that shared
    function is wired to a retrieval-relevance variant of FEVER
    (``mteb/fever``) for Part 2's evidence-retrieval use and carries no
    verdict labels, so it can't serve Part 3's NLI evaluation.
    """
    raw = load_dataset("copenlu/fever_gold_evidence")

    def cast(ds: Dataset) -> Dataset:
        cols = ds.column_names
        ds = ds.map(_fever_gold_to_nli, remove_columns=cols)
        return _clean(ds)

    return DatasetDict({
        "train": cast(raw["train"]),
        "dev": cast(raw["validation"]),
        "test": cast(raw["test"]),
    })


def _cap_train(train: Dataset, max_train_examples: Optional[int], seed: int) -> Dataset:
    """Shuffle and cap a train split to at most ``max_train_examples`` rows."""
    if not max_train_examples or len(train) <= max_train_examples:
        return train
    return train.shuffle(seed=seed).select(range(max_train_examples))


def load_nli(
    source: Literal["mnli", "anli", "snli", "fever", "mnli+anli", "mnli+anli+fever"] = "mnli",
    anli_round: Literal[1, 2, 3] = 1,
    max_train_examples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """Return (train, dev, test_or_none) in the unified schema.

    ``mnli+anli`` concatenates MNLI train with all three ANLI rounds so the
    model sees both distributions during fine-tuning (the standard recipe
    from the Adversarial NLI paper).

    ``max_train_examples`` caps (after shuffling) the train split only,
    for every source -- used for quick smoke tests.
    """
    if source == "mnli":
        d = load_mnli()
        train = _cap_train(d["train"], max_train_examples, seed)
        return train, d["validation_matched"], d["validation_mismatched"]

    if source == "anli":
        d = load_anli(anli_round)
        train = _cap_train(d["train"], max_train_examples, seed)
        return train, d["dev"], d["test"]

    if source == "snli":
        d = load_snli()
        train = _cap_train(d["train"], max_train_examples, seed)
        return train, d["validation"], d["test"]

    if source == "fever":
        d = load_fever_as_nli()
        train = _cap_train(d["train"], max_train_examples, seed)
        return train, d["dev"], d["test"]

    if source == "mnli+anli":
        mnli = load_mnli()
        parts = [mnli["train"]]
        for r in (1, 2, 3):
            parts.append(load_anli(r)["train"])
        train = concatenate_datasets(parts).shuffle(seed=seed)
        train = _cap_train(train, max_train_examples, seed)
        return train, mnli["validation_matched"], mnli["validation_mismatched"]

    if source == "mnli+anli+fever":
        mnli = load_mnli()
        parts = [mnli["train"]]
        for r in (1, 2, 3):
            parts.append(load_anli(r)["train"])
        parts.append(load_fever_as_nli()["train"])
        train = concatenate_datasets(parts).shuffle(seed=seed)
        train = _cap_train(train, max_train_examples, seed)
        return train, mnli["validation_matched"], mnli["validation_mismatched"]

    raise ValueError(f"Unknown source: {source}")


def label_distribution(ds: Dataset) -> Dict[int, int]:
    """Small helper used in reports."""
    from collections import Counter
    return dict(Counter(ds["label"]))
