#!/usr/bin/env python3
"""Train an aligned shallow CatBoost Plain/Ordered candidate for S6E2."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (SRC_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from core.cv_folds import load_or_create_fold_artifact
from train_v7_realmlp import (
    ID_COLUMN,
    POSITIVE_LABEL,
    TARGET,
    add_bin_digit_features,
    add_original_statistics,
    load_original_rows,
    parse_folds_to_run,
    prepare_representation,
    validate_competition_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    original_source = parser.add_mutually_exclusive_group(required=True)
    original_source.add_argument("--combined", type=Path)
    original_source.add_argument("--original", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds-to-run", default="all")
    parser.add_argument(
        "--representation",
        choices=("all_categorical", "hybrid"),
        default="all_categorical",
    )
    parser.add_argument("--hybrid-cardinality", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        choices=("base", "bin_digit"),
        default="base",
    )
    parser.add_argument(
        "--original-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--original-smoothing", type=float, default=10.0)
    parser.add_argument(
        "--original-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--boosting-type", choices=("Ordered", "Plain"), default="Ordered")
    parser.add_argument("--depth", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--task-type", choices=("auto", "CPU", "GPU"), default="auto")
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_task_type(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_file = args.fold_file or args.output_dir / "canonical_folds.npz"
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    base_features = validate_competition_frames(train, test)
    y = train[TARGET].eq(POSITIVE_LABEL).astype("int8").to_numpy()
    fold_ids = load_or_create_fold_artifact(
        fold_file,
        ids=train[ID_COLUMN].to_numpy(),
        target=y,
        n_splits=args.folds,
        seed=args.seed,
    )
    if args.original_stats:
        original = load_original_rows(
            combined_path=args.combined,
            original_path=args.original,
            features=base_features,
        )
        X, X_test, original_stat_features = add_original_statistics(
            train[base_features],
            test[base_features],
            original,
            base_features=base_features,
            smoothing=args.original_smoothing,
            include_pairs=args.original_pairs,
        )
    else:
        X = train[base_features].copy()
        X_test = test[base_features].copy()
        original_stat_features = []

    engineered_features: list[str] = []
    if args.feature_set == "bin_digit":
        X, X_test, engineered_features = add_bin_digit_features(
            X,
            X_test,
            base_features=base_features,
        )
    representation_features = base_features + engineered_features
    X, X_test, categorical_features = prepare_representation(
        X,
        X_test,
        base_features=representation_features,
        representation=args.representation,
        hybrid_cardinality=args.hybrid_cardinality,
    )
    folds_to_run = parse_folds_to_run(args.folds_to_run, args.folds)
    checkpoints = args.output_dir / "fold_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    task_type = resolve_task_type(args.task_type)

    print(
        f"rows={len(train):,} features={X.shape[1]} categorical={len(categorical_features)} "
        f"task_type={task_type} boosting={args.boosting_type} depth={args.depth}"
    )
    for fold in folds_to_run:
        checkpoint = checkpoints / f"fold_{fold}.npz"
        if args.resume and checkpoint.exists():
            print(f"fold={fold} checkpoint_exists=1 action=skip")
            continue
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        start = time.time()

        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=args.l2_leaf_reg,
            random_strength=args.random_strength,
            boosting_type=args.boosting_type,
            bootstrap_type="Bernoulli",
            subsample=args.subsample,
            random_seed=args.seed + 100 * fold,
            task_type=task_type,
            verbose=args.verbose,
            allow_writing_files=False,
        )
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            cat_features=categorical_features,
            eval_set=(X.iloc[valid_idx], y[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        valid_prediction = model.predict_proba(X.iloc[valid_idx])[:, 1]
        test_prediction = model.predict_proba(X_test)[:, 1]
        fold_auc = float(roc_auc_score(y[valid_idx], valid_prediction))
        np.savez_compressed(
            checkpoint,
            valid_idx=valid_idx,
            valid_prediction=valid_prediction,
            test_prediction=test_prediction,
            fold_auc=np.asarray(fold_auc),
            best_iteration=np.asarray(model.get_best_iteration()),
            elapsed_seconds=np.asarray(time.time() - start),
        )
        print(
            f"fold={fold} auc={fold_auc:.9f} best_iteration={model.get_best_iteration()} "
            f"elapsed_seconds={time.time() - start:.1f}"
        )
        del model
        gc.collect()

    missing = [
        fold for fold in range(args.folds) if not (checkpoints / f"fold_{fold}.npz").exists()
    ]
    if missing:
        print(f"run_incomplete=1 missing_folds={missing}")
        return

    oof = np.zeros(len(train), dtype="float64")
    test_predictions: list[np.ndarray] = []
    fold_auc: list[float] = []
    best_iterations: list[int] = []
    elapsed_seconds: list[float] = []
    for fold in range(args.folds):
        artifact = np.load(checkpoints / f"fold_{fold}.npz")
        valid_idx = artifact["valid_idx"]
        if not np.array_equal(valid_idx, np.flatnonzero(fold_ids == fold)):
            raise ValueError(f"Fold {fold} checkpoint indices do not match protocol")
        oof[valid_idx] = artifact["valid_prediction"]
        test_predictions.append(artifact["test_prediction"])
        fold_auc.append(float(artifact["fold_auc"]))
        best_iterations.append(int(artifact["best_iteration"]))
        elapsed_seconds.append(float(artifact["elapsed_seconds"]))
    test_prediction = np.mean(test_predictions, axis=0)
    overall_auc = float(roc_auc_score(y, oof))

    stem = f"catboost_{args.boosting_type.lower()}_d{args.depth}_{args.representation}"
    if not args.original_stats:
        stem = f"{stem}_raw"
    if args.feature_set != "base":
        stem = f"{stem}_{args.feature_set}"
    np.savez_compressed(
        args.output_dir / f"{stem}_oof.npz",
        id=train[ID_COLUMN].to_numpy(),
        target=y,
        prediction=oof,
        fold=fold_ids,
    )
    np.save(args.output_dir / f"{stem}_test.npy", test_prediction)
    submission = test[[ID_COLUMN]].copy()
    submission[TARGET] = test_prediction
    submission.to_csv(args.output_dir / f"submission_{stem}.csv", index=False)
    metrics = {
        "model": "CatBoostClassifier",
        "boosting_type": args.boosting_type,
        "depth": args.depth,
        "representation": args.representation,
        "feature_set": args.feature_set,
        "original_stats": args.original_stats,
        "overall_oof_auc": overall_auc,
        "fold_auc": fold_auc,
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc, ddof=1)),
        "best_iteration_by_fold": best_iterations,
        "elapsed_seconds_by_fold": elapsed_seconds,
        "feature_count": int(X.shape[1]),
        "categorical_feature_count": len(categorical_features),
        "original_stat_feature_count": len(original_stat_features),
        "engineered_feature_count": len(engineered_features),
        "task_type": task_type,
        "folds": args.folds,
        "seed": args.seed,
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
