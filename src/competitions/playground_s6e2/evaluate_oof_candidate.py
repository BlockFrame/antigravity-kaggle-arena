#!/usr/bin/env python3
"""Evaluate one aligned OOF candidate against a base blend without submission."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--base-test", type=Path)
    parser.add_argument("--candidate-test", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_oof(path: Path) -> dict[str, np.ndarray]:
    artifact = np.load(path)
    required = {"id", "target", "prediction", "fold"}
    if missing := required.difference(artifact.files):
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    result = {name: artifact[name] for name in required}
    prediction = result["prediction"]
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise ValueError(f"{path} contains invalid predictions")
    return result


def normalized_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite one-dimensional arrays")
    return rankdata(values, method="average") / len(values)


def validate_alignment(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> None:
    for key in ("id", "target", "fold"):
        if not np.array_equal(base[key], candidate[key]):
            raise ValueError(f"OOF artifacts do not align on {key!r}")


def weight_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("--grid-step must be in (0, 1]")
    count = int(round(1.0 / step))
    if not np.isclose(count * step, 1.0):
        raise ValueError("--grid-step must divide 1.0 exactly")
    return np.linspace(0.0, 1.0, count + 1)


def blend(
    base_rank: np.ndarray,
    candidate_rank: np.ndarray,
    candidate_weight: float,
) -> np.ndarray:
    return (1.0 - candidate_weight) * base_rank + candidate_weight * candidate_rank


def select_weight(
    y: np.ndarray,
    base_rank: np.ndarray,
    candidate_rank: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    scored = [
        (
            float(weight),
            float(
                roc_auc_score(
                    y[mask],
                    blend(base_rank, candidate_rank, float(weight))[mask],
                )
            ),
        )
        for weight in weights
    ]
    # Prefer the smaller candidate weight when scores tie at machine precision.
    return max(scored, key=lambda item: (item[1], -item[0]))


def evaluate(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    step: float,
) -> tuple[dict[str, object], float]:
    validate_alignment(base, candidate)
    y = base["target"]
    folds = base["fold"]
    base_rank = normalized_rank(base["prediction"])
    candidate_rank = normalized_rank(candidate["prediction"])
    weights = weight_grid(step)
    all_rows = np.ones(len(y), dtype=bool)
    full_weight, full_auc = select_weight(
        y,
        base_rank,
        candidate_rank,
        weights,
        all_rows,
    )

    crossfit_prediction = np.zeros(len(y), dtype="float64")
    held_out_rows: list[dict[str, float | int]] = []
    selected_weights: list[float] = []
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        selected_weight, selection_auc = select_weight(
            y,
            base_rank,
            candidate_rank,
            weights,
            train_mask,
        )
        selected_weights.append(selected_weight)
        fold_prediction = blend(base_rank, candidate_rank, selected_weight)
        crossfit_prediction[valid_mask] = fold_prediction[valid_mask]
        base_fold_auc = float(roc_auc_score(y[valid_mask], base_rank[valid_mask]))
        candidate_fold_auc = float(
            roc_auc_score(y[valid_mask], candidate_rank[valid_mask])
        )
        blend_fold_auc = float(
            roc_auc_score(y[valid_mask], fold_prediction[valid_mask])
        )
        held_out_rows.append(
            {
                "fold": int(fold),
                "selected_candidate_weight": selected_weight,
                "selection_auc_on_other_folds": selection_auc,
                "base_auc": base_fold_auc,
                "candidate_auc": candidate_fold_auc,
                "blend_auc": blend_fold_auc,
                "blend_delta_vs_base": blend_fold_auc - base_fold_auc,
            }
        )

    conservative_weight = float(np.median(selected_weights))
    conservative_prediction = blend(base_rank, candidate_rank, conservative_weight)
    result: dict[str, object] = {
        "base_oof_auc": float(roc_auc_score(y, base_rank)),
        "candidate_oof_auc": float(roc_auc_score(y, candidate_rank)),
        "spearman": float(spearmanr(base_rank, candidate_rank).statistic),
        "full_oof_best_candidate_weight_diagnostic_only": full_weight,
        "full_oof_best_auc_diagnostic_only": full_auc,
        "crossfit_selected_weights": selected_weights,
        "crossfit_oof_auc": float(roc_auc_score(y, crossfit_prediction)),
        "conservative_candidate_weight": conservative_weight,
        "conservative_oof_auc": float(roc_auc_score(y, conservative_prediction)),
        "conservative_delta_vs_base": float(
            roc_auc_score(y, conservative_prediction) - roc_auc_score(y, base_rank)
        ),
        "held_out_fold_results": held_out_rows,
    }
    return result, conservative_weight


def maybe_write_submission(
    args: argparse.Namespace,
    candidate_weight: float,
) -> Path | None:
    supplied = [args.base_test, args.candidate_test, args.test_csv, args.output_dir]
    if not any(item is not None for item in supplied):
        return None
    if not all(item is not None for item in supplied):
        raise ValueError(
            "Submission generation requires --base-test, --candidate-test, "
            "--test-csv, and --output-dir together"
        )
    base_test = np.load(args.base_test)
    candidate_test = np.load(args.candidate_test)
    test = pd.read_csv(args.test_csv)
    if len(base_test) != len(test) or len(candidate_test) != len(test):
        raise ValueError("Test prediction lengths do not match test.csv")
    prediction = blend(
        normalized_rank(base_test),
        normalized_rank(candidate_test),
        candidate_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "submission_candidate_rank_blend.csv"
    submission = test[[ID_COLUMN]].copy()
    submission[TARGET] = prediction
    submission.to_csv(path, index=False)
    np.save(args.output_dir / "candidate_rank_blend_test.npy", prediction)
    return path


def main() -> None:
    args = parse_args()
    base = load_oof(args.base_oof)
    candidate = load_oof(args.candidate_oof)
    result, conservative_weight = evaluate(base, candidate, step=args.grid_step)
    submission_path = maybe_write_submission(args, conservative_weight)
    if submission_path is not None:
        result["local_submission"] = str(submission_path.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
