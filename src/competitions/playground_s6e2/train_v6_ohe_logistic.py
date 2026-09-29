#!/usr/bin/env python3
"""Train a leakage-safe one-hot logistic baseline for Playground S6E2.

All competition features are treated as categorical.  The optional 303-row
original dataset is appended to each fold's training partition, but it is
never included in validation.  This keeps the OOF score aligned with the
630,000-row synthetic competition training distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.cv_folds import load_or_create_fold_artifact


TARGET = "Heart Disease"
POSITIVE_LABEL = "Presence"
ID_COLUMN = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--combined",
        type=Path,
        help=(
            "Optional train_combined.csv containing an is_generated column. "
            "Rows where is_generated == 0 are used as original-data anchors."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, dest="regularization_c")
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def _validate_inputs(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    required_train = {ID_COLUMN, TARGET}
    required_test = {ID_COLUMN}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"Train is missing required columns: {sorted(missing)}")
    if missing := required_test.difference(test.columns):
        raise ValueError(f"Test is missing required columns: {sorted(missing)}")

    features = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if not features:
        raise ValueError("No feature columns found")
    if features != [c for c in test.columns if c != ID_COLUMN]:
        raise ValueError("Train and test feature columns/order do not match")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate ids detected")
    return features


def _as_categories(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return stable string categories without mutating the input frame."""
    result = frame.loc[:, features].copy()
    for column in features:
        result[column] = result[column].astype("string").fillna("__MISSING__")
    return result


def _load_original_rows(combined_path: Path | None, features: list[str]) -> pd.DataFrame:
    if combined_path is None:
        return pd.DataFrame(columns=features)

    combined = pd.read_csv(combined_path)
    if "is_generated" not in combined.columns:
        raise ValueError("--combined must contain an is_generated column")
    if missing := set(features).difference(combined.columns):
        raise ValueError(f"Combined data is missing features: {sorted(missing)}")

    original = combined.loc[combined["is_generated"].eq(0)].copy()
    if TARGET not in original.columns:
        raise ValueError(f"Combined data is missing target column {TARGET!r}")
    if original.empty:
        raise ValueError("Combined data contains no rows with is_generated == 0")
    return original


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.regularization_c <= 0:
        raise ValueError("--c must be positive")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    features = _validate_inputs(train, test)
    original = _load_original_rows(args.combined, features)

    X = _as_categories(train, features)
    X_test = _as_categories(test, features)
    y = train[TARGET].eq(POSITIVE_LABEL).astype("int8").to_numpy()

    if original.empty:
        X_original = None
        y_original = None
    else:
        X_original = _as_categories(original, features)
        y_original = original[TARGET].eq(POSITIVE_LABEL).astype("int8").to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_file = args.fold_file or args.output_dir / "canonical_folds.npz"
    fold_ids = load_or_create_fold_artifact(
        fold_file,
        ids=train[ID_COLUMN].to_numpy(),
        target=y,
        n_splits=args.folds,
        seed=args.seed,
    )
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    encoded_feature_counts: list[int] = []

    for fold in range(args.folds):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y[train_idx]
        if X_original is not None and y_original is not None:
            X_fold_train = pd.concat([X_fold_train, X_original], ignore_index=True)
            y_fold_train = np.concatenate([y_fold_train, y_original])

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            dtype=np.float32,
            sparse_output=True,
        )
        X_train_encoded = encoder.fit_transform(X_fold_train)
        X_valid_encoded = encoder.transform(X.iloc[valid_idx])
        X_test_encoded = encoder.transform(X_test)

        model = LogisticRegression(
            C=args.regularization_c,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.seed + fold,
        )
        model.fit(X_train_encoded, y_fold_train)
        valid_prediction = model.predict_proba(X_valid_encoded)[:, 1]
        oof[valid_idx] = valid_prediction
        test_prediction += model.predict_proba(X_test_encoded)[:, 1] / args.folds

        fold_auc = float(roc_auc_score(y[valid_idx], valid_prediction))
        fold_scores.append(fold_auc)
        encoded_feature_counts.append(int(X_train_encoded.shape[1]))
        print(
            f"fold={fold + 1}/{args.folds} auc={fold_auc:.9f} "
            f"encoded_features={X_train_encoded.shape[1]}"
        )

    overall_auc = float(roc_auc_score(y, oof))

    np.savez_compressed(
        args.output_dir / "ohe_logistic_oof.npz",
        id=train[ID_COLUMN].to_numpy(),
        target=y,
        prediction=oof,
        fold=fold_ids,
    )
    np.save(args.output_dir / "ohe_logistic_test.npy", test_prediction)

    submission = test[[ID_COLUMN]].copy()
    submission[TARGET] = test_prediction
    submission.to_csv(args.output_dir / "submission_ohe_logistic.csv", index=False)

    metrics = {
        "model": "one_hot_logistic_regression",
        "overall_oof_auc": overall_auc,
        "fold_auc": fold_scores,
        "fold_auc_mean": float(np.mean(fold_scores)),
        "fold_auc_std": float(np.std(fold_scores, ddof=1)),
        "competition_train_rows": int(len(train)),
        "original_anchor_rows": int(len(original)),
        "test_rows": int(len(test)),
        "feature_count": len(features),
        "encoded_feature_count_by_fold": encoded_feature_counts,
        "folds": args.folds,
        "seed": args.seed,
        "regularization_c": args.regularization_c,
        "max_iter": args.max_iter,
    }
    (args.output_dir / "ohe_logistic_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"overall_oof_auc={overall_auc:.9f}")
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
