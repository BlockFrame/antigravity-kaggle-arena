#!/usr/bin/env python3
"""Build the conservative V6 rank blend from V5 GBDTs and OHE logistic.

This script deliberately uses a fixed 50/50 family-level blend.  It avoids an
extra optimizer over the same OOF labels and sits in the broad stable region
observed during the audit (40% to 60% logistic weight).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


TARGET = "Heart Disease"
ID_COLUMN = "id"
V5_WEIGHTS = {"cat": 0.615, "lgb": 0.204, "xgb": 0.181}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--logistic-oof", type=Path, required=True)
    parser.add_argument("--logistic-test", type=Path, required=True)
    parser.add_argument("--v5-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logistic-weight", type=float, default=0.5)
    return parser.parse_args()


def normalized_rank(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite one-dimensional arrays")
    return rankdata(values, method="average") / len(values)


def load_legacy_generated_oof(
    path: Path,
    generated_mask: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    values = np.load(path)
    if len(values) != len(generated_mask):
        raise ValueError(
            f"{path.name}: expected {len(generated_mask)} combined rows, got {len(values)}"
        )
    generated = values[generated_mask]
    if len(generated) != expected_rows:
        raise ValueError(f"{path.name}: generated-row count does not match train")
    return generated


def verify_combined_alignment(
    train: pd.DataFrame,
    combined: pd.DataFrame,
    generated_mask: np.ndarray,
) -> None:
    features = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if missing := set(features).difference(combined.columns):
        raise ValueError(f"Combined data is missing features: {sorted(missing)}")
    generated = combined.loc[generated_mask, features].reset_index(drop=True)
    expected = train.loc[:, features].reset_index(drop=True)
    # CSV type inference may represent an integer-looking feature as float in
    # one file and integer in the other.  Compare values column-wise while
    # still rejecting any row/order mismatch.
    for column in features:
        left = generated[column].to_numpy()
        right = expected[column].to_numpy()
        equal = (left == right) | (pd.isna(left) & pd.isna(right))
        if not bool(np.all(equal)):
            raise ValueError(
                "Generated rows in combined data are not aligned with "
                f"train.csv (first mismatch in {column!r})"
            )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.logistic_weight <= 1.0:
        raise ValueError("--logistic-weight must be between 0 and 1")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    combined = pd.read_csv(args.combined)
    if "is_generated" not in combined.columns:
        raise ValueError("Combined data must contain is_generated")
    generated_mask = combined["is_generated"].eq(1).to_numpy()
    verify_combined_alignment(train, combined, generated_mask)

    logistic_bundle = np.load(args.logistic_oof)
    logistic_oof = logistic_bundle["prediction"]
    y = logistic_bundle["target"]
    folds = logistic_bundle["fold"]
    logistic_ids = logistic_bundle["id"]
    if not np.array_equal(logistic_ids, train[ID_COLUMN].to_numpy()):
        raise ValueError("Logistic OOF ids do not align with train.csv")

    v5_oof: dict[str, np.ndarray] = {}
    v5_test: dict[str, np.ndarray] = {}
    for family in V5_WEIGHTS:
        v5_oof[family] = load_legacy_generated_oof(
            args.v5_oof_dir / f"oof_v5_leakfree_{family}.npy",
            generated_mask,
            len(train),
        )
        v5_test[family] = np.load(args.v5_oof_dir / f"test_preds_v5_{family}.npy")
        if len(v5_test[family]) != len(test):
            raise ValueError(f"V5 {family} test predictions do not align with test.csv")

    logistic_test = np.load(args.logistic_test)
    if len(logistic_oof) != len(train) or len(logistic_test) != len(test):
        raise ValueError("Logistic prediction lengths do not match train/test")

    v5_oof_rank = sum(V5_WEIGHTS[k] * normalized_rank(v) for k, v in v5_oof.items())
    v5_test_rank = sum(V5_WEIGHTS[k] * normalized_rank(v) for k, v in v5_test.items())
    logistic_oof_rank = normalized_rank(logistic_oof)
    logistic_test_rank = normalized_rank(logistic_test)

    weight = args.logistic_weight
    blend_oof = weight * logistic_oof_rank + (1.0 - weight) * v5_oof_rank
    blend_test = weight * logistic_test_rank + (1.0 - weight) * v5_test_rank

    fold_auc = [
        float(roc_auc_score(y[folds == fold], blend_oof[folds == fold]))
        for fold in sorted(np.unique(folds))
    ]
    metrics = {
        "model": "v6_fixed_rank_blend",
        "overall_oof_auc": float(roc_auc_score(y, blend_oof)),
        "fold_auc": fold_auc,
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc, ddof=1)),
        "logistic_oof_auc": float(roc_auc_score(y, logistic_oof)),
        "v5_reconstructed_oof_auc": float(roc_auc_score(y, v5_oof_rank)),
        "logistic_weight": weight,
        "v5_family_weights": V5_WEIGHTS,
        "logistic_v5_spearman": float(
            spearmanr(logistic_oof_rank, v5_oof_rank).statistic
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "v6_rank_blend_oof.npz",
        id=train[ID_COLUMN].to_numpy(),
        target=y,
        prediction=blend_oof,
        fold=folds,
    )
    np.save(args.output_dir / "v6_rank_blend_test.npy", blend_test)
    submission = test[[ID_COLUMN]].copy()
    submission[TARGET] = blend_test
    submission.to_csv(args.output_dir / "submission_v6_rank_blend.csv", index=False)
    (args.output_dir / "v6_rank_blend_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
