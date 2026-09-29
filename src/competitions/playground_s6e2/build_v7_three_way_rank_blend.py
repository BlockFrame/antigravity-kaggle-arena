#!/usr/bin/env python3
"""Build a cross-fitted V6 + RealMLP + CatBoost rank blend for S6E2."""

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
REQUIRED_OOF_ARRAYS = ("id", "target", "prediction", "fold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-oof", type=Path, required=True)
    parser.add_argument("--realmlp-oof", type=Path, required=True)
    parser.add_argument("--catboost-oof", type=Path, required=True)
    parser.add_argument("--v6-test", type=Path, required=True)
    parser.add_argument("--realmlp-test", type=Path, required=True)
    parser.add_argument("--catboost-test", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--realmlp-weight", type=float, default=0.70)
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser.parse_args()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as artifact:
        missing = set(REQUIRED_OOF_ARRAYS).difference(artifact.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        result = {name: artifact[name] for name in REQUIRED_OOF_ARRAYS}
    prediction = result["prediction"]
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise ValueError(f"{path} contains invalid predictions")
    return result


def normalized_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions must be a finite one-dimensional array")
    return rankdata(values, method="average") / len(values)


def validate_alignment(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    name: str,
) -> None:
    for key in ("id", "target", "fold"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{name} does not align with V6 on {key!r}")


def weight_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("--grid-step must be in (0, 1]")
    count = int(round(1.0 / step))
    if not np.isclose(count * step, 1.0):
        raise ValueError("--grid-step must divide 1.0 exactly")
    return np.linspace(0.0, 1.0, count + 1)


def select_catboost_weight(
    y: np.ndarray,
    base_rank: np.ndarray,
    catboost_rank: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    scored = []
    for weight in weights:
        prediction = (1.0 - weight) * base_rank + weight * catboost_rank
        scored.append((float(weight), float(roc_auc_score(y[mask], prediction[mask]))))
    return max(scored, key=lambda item: (item[1], -item[0]))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.realmlp_weight <= 1.0:
        raise ValueError("--realmlp-weight must be in [0, 1]")

    v6 = load_oof(args.v6_oof)
    realmlp = load_oof(args.realmlp_oof)
    catboost = load_oof(args.catboost_oof)
    validate_alignment(v6, realmlp, "RealMLP")
    validate_alignment(v6, catboost, "CatBoost")

    y = v6["target"]
    folds = v6["fold"]
    v6_rank = normalized_rank(v6["prediction"])
    realmlp_rank = normalized_rank(realmlp["prediction"])
    catboost_rank = normalized_rank(catboost["prediction"])
    base_rank = normalized_rank(
        (1.0 - args.realmlp_weight) * v6_rank
        + args.realmlp_weight * realmlp_rank
    )
    weights = weight_grid(args.grid_step)

    crossfit_prediction = np.zeros(len(y), dtype="float64")
    selected_weights: list[float] = []
    held_out_results: list[dict[str, float | int]] = []
    for fold in sorted(np.unique(folds)):
        selection_mask = folds != fold
        held_out_mask = folds == fold
        weight, selection_auc = select_catboost_weight(
            y, base_rank, catboost_rank, selection_mask, weights
        )
        selected_weights.append(weight)
        fold_prediction = (1.0 - weight) * base_rank + weight * catboost_rank
        crossfit_prediction[held_out_mask] = fold_prediction[held_out_mask]
        base_auc = float(roc_auc_score(y[held_out_mask], base_rank[held_out_mask]))
        blend_auc = float(
            roc_auc_score(y[held_out_mask], fold_prediction[held_out_mask])
        )
        held_out_results.append(
            {
                "fold": int(fold),
                "selected_catboost_weight": weight,
                "selection_auc_on_other_folds": selection_auc,
                "base_auc": base_auc,
                "blend_auc": blend_auc,
                "blend_delta_vs_base": blend_auc - base_auc,
            }
        )

    conservative_catboost_weight = float(np.median(selected_weights))
    conservative_prediction = (
        (1.0 - conservative_catboost_weight) * base_rank
        + conservative_catboost_weight * catboost_rank
    )
    base_auc = float(roc_auc_score(y, base_rank))
    crossfit_auc = float(roc_auc_score(y, crossfit_prediction))
    conservative_auc = float(roc_auc_score(y, conservative_prediction))

    v6_weight = (1.0 - conservative_catboost_weight) * (
        1.0 - args.realmlp_weight
    )
    realmlp_weight = (
        1.0 - conservative_catboost_weight
    ) * args.realmlp_weight

    test = pd.read_csv(args.test_csv)
    test_predictions = {
        "v6": np.load(args.v6_test),
        "realmlp": np.load(args.realmlp_test),
        "catboost": np.load(args.catboost_test),
    }
    for name, prediction in test_predictions.items():
        if prediction.ndim != 1 or len(prediction) != len(test):
            raise ValueError(f"{name} test predictions do not align with test.csv")
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} test predictions contain non-finite values")

    base_test_rank = normalized_rank(
        (1.0 - args.realmlp_weight) * normalized_rank(test_predictions["v6"])
        + args.realmlp_weight * normalized_rank(test_predictions["realmlp"])
    )
    final_test_prediction = (
        (1.0 - conservative_catboost_weight) * base_test_rank
        + conservative_catboost_weight * normalized_rank(test_predictions["catboost"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "v7_three_way_rank_blend"
    np.savez_compressed(
        args.output_dir / f"{stem}_oof.npz",
        id=v6["id"],
        target=y,
        prediction=crossfit_prediction,
        conservative_prediction=conservative_prediction,
        fold=folds,
    )
    np.save(args.output_dir / f"{stem}_test.npy", final_test_prediction)
    submission = test[[ID_COLUMN]].copy()
    submission[TARGET] = final_test_prediction
    submission_path = args.output_dir / f"submission_{stem}.csv"
    submission.to_csv(submission_path, index=False)

    metrics = {
        "model": "rank_blend",
        "components": ["v6_rank_blend", "realmlp", "catboost_ordered_d3"],
        "component_weights": {
            "v6_rank_blend": v6_weight,
            "realmlp": realmlp_weight,
            "catboost_ordered_d3": conservative_catboost_weight,
        },
        "selected_catboost_weights_by_fold": selected_weights,
        "base_v6_realmlp_oof_auc": base_auc,
        "crossfit_oof_auc": crossfit_auc,
        "crossfit_delta_vs_base": crossfit_auc - base_auc,
        "conservative_oof_auc": conservative_auc,
        "conservative_delta_vs_base": conservative_auc - base_auc,
        "spearman_base_vs_catboost": float(
            spearmanr(base_rank, catboost_rank).statistic
        ),
        "held_out_fold_results": held_out_results,
        "submission": str(submission_path.resolve()),
    }
    metrics_path = args.output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
