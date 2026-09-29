"""Canonical cross-validation fold artifacts shared by competition pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_stratified_fold_ids(
    target: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return one deterministic validation-fold id for every training row."""
    target = np.asarray(target)
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(target) < n_splits:
        raise ValueError("target has fewer rows than n_splits")

    fold_ids = np.full(len(target), -1, dtype=np.int16)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    placeholder = np.zeros(len(target), dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splitter.split(placeholder, target)):
        fold_ids[valid_idx] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned to a validation fold")
    return fold_ids


def save_fold_artifact(
    path: Path,
    *,
    ids: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    n_splits: int,
    seed: int,
) -> None:
    """Persist ids, labels, and folds together so alignment can be audited."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        id=np.asarray(ids),
        target=np.asarray(target),
        fold=np.asarray(fold_ids),
        n_splits=np.asarray(n_splits),
        seed=np.asarray(seed),
    )


def load_or_create_fold_artifact(
    path: Path,
    *,
    ids: np.ndarray,
    target: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Load a fold artifact after strict alignment checks, or create it."""
    ids = np.asarray(ids)
    target = np.asarray(target)
    if path.exists():
        artifact = np.load(path)
        stored_ids = artifact["id"]
        stored_target = artifact["target"]
        fold_ids = artifact["fold"]
        stored_n_splits = int(artifact["n_splits"])
        stored_seed = int(artifact["seed"])
        if not np.array_equal(stored_ids, ids):
            raise ValueError(f"Fold artifact ids do not align: {path}")
        if not np.array_equal(stored_target, target):
            raise ValueError(f"Fold artifact targets do not align: {path}")
        if stored_n_splits != n_splits or stored_seed != seed:
            raise ValueError(
                "Fold artifact protocol mismatch: "
                f"stored n_splits={stored_n_splits}, seed={stored_seed}; "
                f"requested n_splits={n_splits}, seed={seed}"
            )
        if len(fold_ids) != len(target) or (fold_ids < 0).any():
            raise ValueError(f"Invalid fold assignments in {path}")
        return fold_ids.astype(np.int16, copy=False)

    fold_ids = make_stratified_fold_ids(target, n_splits=n_splits, seed=seed)
    save_fold_artifact(
        path,
        ids=ids,
        target=target,
        fold_ids=fold_ids,
        n_splits=n_splits,
        seed=seed,
    )
    return fold_ids
