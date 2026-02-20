from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def build_repeated_stratified_splits(
    y_event: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build repeated stratified train/validation splits from event labels."""
    y = np.asarray(y_event).astype(int)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Need both event and censoring classes for stratified CV")
    if counts.min() < 2:
        raise ValueError("Insufficient minority class count for stratified CV")
    effective_splits = min(n_splits, int(counts.min()))
    if effective_splits < 2:
        raise ValueError("effective stratified folds < 2")

    rkf = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    idx = np.arange(len(y))
    return [(tr, va) for tr, va in rkf.split(idx, y)]
