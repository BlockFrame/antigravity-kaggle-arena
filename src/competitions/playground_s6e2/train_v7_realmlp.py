#!/usr/bin/env python3
"""Train the V7 RealMLP portfolio member for Playground S6E2.

The script uses canonical outer folds on the 630,000 synthetic competition
rows. Labels from the 303-row original dataset are used only to derive fixed
smoothed statistics; original rows are never placed in an outer validation
fold. Each completed fold is checkpointed so GPU jobs can resume safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from core.cv_folds import load_or_create_fold_artifact
except ModuleNotFoundError:  # Allows this file to run as a standalone Kaggle script.
    from sklearn.model_selection import StratifiedKFold

    def load_or_create_fold_artifact(
        path: Path,
        *,
        ids: np.ndarray,
        target: np.ndarray,
        n_splits: int = 5,
        seed: int = 42,
    ) -> np.ndarray:
        ids = np.asarray(ids)
        target = np.asarray(target)
        if path.exists():
            artifact = np.load(path)
            if not np.array_equal(artifact["id"], ids):
                raise ValueError("Fold artifact ids do not align")
            if not np.array_equal(artifact["target"], target):
                raise ValueError("Fold artifact targets do not align")
            if int(artifact["n_splits"]) != n_splits or int(artifact["seed"]) != seed:
                raise ValueError("Fold artifact protocol mismatch")
            return artifact["fold"].astype(np.int16, copy=False)

        fold_ids = np.full(len(target), -1, dtype=np.int16)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (_, valid_idx) in enumerate(
            splitter.split(np.zeros(len(target), dtype=np.int8), target)
        ):
            fold_ids[valid_idx] = fold
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            id=ids,
            target=target,
            fold=fold_ids,
            n_splits=np.asarray(n_splits),
            seed=np.asarray(seed),
        )
        return fold_ids


TARGET = "Heart Disease"
POSITIVE_LABEL = "Presence"
ID_COLUMN = "id"
DEFAULT_PAIRS = (
    ("Sex", "Chest pain type"),
    ("Chest pain type", "Thallium"),
    ("Slope of ST", "Exercise angina"),
    ("Number of vessels fluro", "Thallium"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("/kaggle/input/playground-series-s6e2/train.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("/kaggle/input/playground-series-s6e2/test.csv"),
    )
    original_source = parser.add_mutually_exclusive_group()
    original_source.add_argument("--combined", type=Path)
    original_source.add_argument(
        "--original",
        type=Path,
        help="Raw 303-row original dataset with the same feature/target columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/v14_realmlp_bin_digit"),
    )
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--folds-to-run",
        default=os.getenv("S6E2_FOLDS_TO_RUN", "all"),
        help="Comma-separated zero-based fold ids, or 'all'.",
    )
    parser.add_argument(
        "--representation",
        choices=("all_categorical", "hybrid"),
        default="all_categorical",
    )
    parser.add_argument("--hybrid-cardinality", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        choices=("base", "bin_digit"),
        default="bin_digit",
        help="Unsupervised feature representation applied before RealMLP.",
    )
    parser.add_argument(
        "--original-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add target statistics derived only from the original dataset.",
    )
    parser.add_argument("--original-smoothing", type=float, default=10.0)
    parser.add_argument(
        "--original-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=Path("/kaggle").exists(),
        help="Fail fast when CUDA is unavailable (enabled by default on Kaggle).",
    )
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-cv", type=int, default=2)
    parser.add_argument("--n-ens", type=int, default=8)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.combined is None and args.original is None:
        args.original = Path(
            "/kaggle/input/s6e4-original-dataset/Heart_Disease_Prediction.csv"
        )
    return args


def validate_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    if TARGET not in train or ID_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError("train/test are missing id or target columns")
    features = [c for c in train.columns if c not in {ID_COLUMN, TARGET}]
    if features != [c for c in test.columns if c != ID_COLUMN]:
        raise ValueError("Train and test feature columns/order do not match")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate ids detected")
    return features


def resolve_input_path(path: Path, *, filename: str, source_hint: str) -> Path:
    """Resolve both legacy and current Kaggle input mount layouts."""
    if path.exists():
        return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = [
            candidate
            for candidate in kaggle_input.rglob(filename)
            if source_hint in str(candidate)
        ]
        if len(candidates) == 1:
            print(f"resolved_input={candidates[0]}")
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous Kaggle input for {filename}: {[str(c) for c in candidates]}"
            )
    raise FileNotFoundError(f"Input file not found: {path}")


def load_original_rows(
    *,
    combined_path: Path | None,
    original_path: Path | None,
    features: list[str],
) -> pd.DataFrame:
    if combined_path is not None:
        combined = pd.read_csv(combined_path)
        required = set(features) | {TARGET, "is_generated"}
        if missing := required.difference(combined.columns):
            raise ValueError(f"Combined data is missing columns: {sorted(missing)}")
        original = combined.loc[
            combined["is_generated"].eq(0), features + [TARGET]
        ].copy()
    elif original_path is not None:
        raw_original = pd.read_csv(original_path)
        required = set(features) | {TARGET}
        if missing := required.difference(raw_original.columns):
            raise ValueError(f"Original data is missing columns: {sorted(missing)}")
        original = raw_original.loc[:, features + [TARGET]].copy()
    else:
        raise ValueError("Either combined_path or original_path is required")
    if original.empty:
        raise ValueError("Original dataset is empty")
    return original


def _safe_name(columns: tuple[str, ...]) -> str:
    return "__".join(re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in columns)


def add_original_statistics(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    original: pd.DataFrame,
    *,
    base_features: list[str],
    smoothing: float,
    include_pairs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Map smoothed target statistics learned only from the original rows."""
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    train_out = train_features.copy()
    test_out = test_features.copy()
    original_target = original[TARGET].eq(POSITIVE_LABEL).astype("float64")
    prior = float(original_target.mean())
    eps = 1e-7
    global_log_odds = math.log((prior + eps) / (1.0 - prior + eps))

    groups: list[tuple[str, ...]] = [(feature,) for feature in base_features]
    if include_pairs:
        groups.extend(pair for pair in DEFAULT_PAIRS if set(pair).issubset(base_features))

    added: list[str] = []
    source = original.loc[:, base_features].copy()
    source["__target__"] = original_target.to_numpy()
    for columns in groups:
        key = list(columns)
        stats = (
            source.groupby(key, dropna=False, observed=False)["__target__"]
            .agg(["count", "sum"])
            .reset_index()
        )
        probability = (stats["sum"] + smoothing * prior) / (
            stats["count"] + smoothing
        )
        prefix = f"orig_{_safe_name(columns)}"
        stat_columns = {
            f"{prefix}_target_mean": probability.astype("float32"),
            f"{prefix}_log_count": np.log1p(stats["count"]).astype("float32"),
            f"{prefix}_woe": (
                np.log((probability + eps) / (1.0 - probability + eps))
                - global_log_odds
            ).astype("float32"),
            f"{prefix}_entropy": (
                -probability * np.log(probability + eps)
                - (1.0 - probability) * np.log(1.0 - probability + eps)
            ).astype("float32"),
        }
        for name, values in stat_columns.items():
            stats[name] = values
        names = list(stat_columns)
        added.extend(names)
        lookup = stats[key + names]
        train_out = train_out.merge(lookup, on=key, how="left", sort=False)
        test_out = test_out.merge(lookup, on=key, how="left", sort=False)
        train_out[names] = train_out[names].fillna(
            {
                names[0]: prior,
                names[1]: 0.0,
                names[2]: 0.0,
                names[3]: -prior * math.log(prior + eps)
                - (1.0 - prior) * math.log(1.0 - prior + eps),
            }
        )
        test_out[names] = test_out[names].fillna(
            {
                names[0]: prior,
                names[1]: 0.0,
                names[2]: 0.0,
                names[3]: -prior * math.log(prior + eps)
                - (1.0 - prior) * math.log(1.0 - prior + eps),
            }
        )

    if len(train_out) != len(train_features) or len(test_out) != len(test_features):
        raise RuntimeError("Original-stat merge changed row counts")
    if train_out[added].isna().any().any() or test_out[added].isna().any().any():
        raise RuntimeError("Original-stat features contain missing values")
    return train_out, test_out, added


def add_bin_digit_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    base_features: list[str],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add label-free binnings and decimal digits for non-binary numeric columns.

    Bin boundaries are learned from competition training features only.  The
    resulting columns are categorical representations, matching the winning
    solution's BASE+BIN+DIGIT+ALL_CATS family without target leakage.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    train_out = train_features.copy()
    test_out = test_features.copy()
    added: list[str] = []

    for column in base_features:
        train_values = pd.to_numeric(train_features[column], errors="raise")
        test_values = pd.to_numeric(test_features[column], errors="raise")
        if train_values.nunique(dropna=True) <= 10:
            continue

        safe = _safe_name((column,))
        finite_train = train_values[np.isfinite(train_values)]
        if finite_train.empty:
            continue

        quantile_edges = np.unique(
            np.quantile(finite_train, np.linspace(0.0, 1.0, n_bins + 1))
        )
        width_edges = np.linspace(
            float(finite_train.min()), float(finite_train.max()), n_bins + 1
        )
        for label, edges in (("qbin", quantile_edges), ("wbin", width_edges)):
            name = f"feat_{safe}_{label}{n_bins}"
            interior = edges[1:-1]
            train_out[name] = np.searchsorted(interior, train_values, side="right")
            test_out[name] = np.searchsorted(interior, test_values, side="right")
            added.append(name)

        # Coarse rounding creates a representation distinct from fixed-width
        # bins because it anchors boundaries at human-readable multiples.
        span = float(finite_train.max() - finite_train.min())
        step = max(1.0, 10.0 ** math.floor(math.log10(max(span / n_bins, 1.0))))
        rounded_name = f"feat_{safe}_round"
        train_out[rounded_name] = np.floor(train_values / step).astype("int32")
        test_out[rounded_name] = np.floor(test_values / step).astype("int32")
        added.append(rounded_name)

        # Integer digits plus the first decimal digit expose generator-like
        # structure while retaining the unmodified source column.
        train_scaled = np.rint(train_values.to_numpy(dtype="float64") * 10).astype("int64")
        test_scaled = np.rint(test_values.to_numpy(dtype="float64") * 10).astype("int64")
        for digit_name, divisor in (("decimal1", 1), ("units", 10), ("tens", 100), ("hundreds", 1000)):
            name = f"feat_{safe}_{digit_name}"
            train_out[name] = (np.abs(train_scaled) // divisor) % 10
            test_out[name] = (np.abs(test_scaled) // divisor) % 10
            added.append(name)

    if train_out[added].isna().any().any() or test_out[added].isna().any().any():
        raise RuntimeError("Bin/digit features contain missing values")
    return train_out, test_out, added


def prepare_representation(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    base_features: list[str],
    representation: str,
    hybrid_cardinality: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train_features.copy()
    test_out = test_features.copy()
    if representation == "all_categorical":
        categorical = list(base_features)
    elif representation == "hybrid":
        categorical = [
            column
            for column in base_features
            if train_out[column].nunique(dropna=False) < hybrid_cardinality
        ]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    for column in categorical:
        categories = pd.Index(
            pd.concat([train_out[column], test_out[column]], ignore_index=True)
            .astype("string")
            .fillna("__MISSING__")
            .unique()
        )
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        train_out[column] = (
            train_out[column].astype("string").fillna("__MISSING__").astype(dtype)
        )
        test_out[column] = (
            test_out[column].astype("string").fillna("__MISSING__").astype(dtype)
        )

    for column in train_out.columns.difference(categorical):
        train_out[column] = pd.to_numeric(train_out[column], errors="raise").astype(
            "float32"
        )
        test_out[column] = pd.to_numeric(test_out[column], errors="raise").astype(
            "float32"
        )
    return train_out, test_out, categorical


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_folds_to_run(value: str, n_splits: int) -> list[int]:
    if value == "all":
        return list(range(n_splits))
    try:
        folds = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as exc:
        raise ValueError("--folds-to-run must be 'all' or comma-separated integers") from exc
    if not folds or any(fold < 0 or fold >= n_splits for fold in folds):
        raise ValueError(f"Fold ids must be between 0 and {n_splits - 1}")
    return folds


def probability_vector(prediction: object) -> np.ndarray:
    values = prediction.to_numpy() if isinstance(prediction, pd.DataFrame) else np.asarray(prediction)
    if values.ndim == 2:
        if values.shape[1] != 2:
            raise ValueError(f"Expected two probability columns, got {values.shape}")
        values = values[:, 1]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions are not a finite one-dimensional vector")
    return values.astype("float64", copy=False)


def realmlp_classifier_class():
    """Import pytabkit, installing the pinned Kaggle runtime dependency if needed."""
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ModuleNotFoundError:
        wheel_candidates = sorted(
            Path("/kaggle/input").glob(
                "**/pytabkit-1.7.3-py3-none-any.whl"
            )
        )
        install_target = (
            str(wheel_candidates[0]) if wheel_candidates else "pytabkit==1.7.3"
        )
        print(f"installing_pytabkit={install_target}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", install_target]
        )
        from pytabkit import RealMLP_TD_Classifier
    return RealMLP_TD_Classifier


def realmlp_parameters(args: argparse.Namespace, fold: int) -> dict[str, object]:
    return {
        "device": resolve_device(args.device),
        "random_state": args.seed + 100 * fold,
        "verbosity": args.verbosity,
        "n_cv": args.n_cv,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "n_ens": args.n_ens,
        "val_metric_name": "1-auc_ovr",
        "use_early_stopping": True,
        "early_stopping_additive_patience": 20,
        "early_stopping_multiplicative_patience": 1,
        "act": "mish",
        "embedding_size": 8,
        "first_layer_lr_factor": 0.5962121993798933,
        "hidden_sizes": "rectangular",
        "hidden_width": 384,
        "lr": 0.04,
        "ls_eps": 0.011498317194338772,
        "ls_eps_sched": "coslog4",
        "max_one_hot_cat_size": 18,
        "n_hidden_layers": 4,
        "p_drop": 0.07301419697186451,
        "p_drop_sched": "flat_cos",
        "use_plr_embeddings": True,
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_lr_factor": 0.1151437622270563,
        "plr_sigma": 2.3316811282666916,
        "scale_lr_factor": 2.244801835541429,
        "sq_mom": 1.0 - 0.011834054955582318,
        "wd": 0.02369230879235962,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.require_gpu and device != "cuda":
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except ImportError:
            torch_version = "missing"
            cuda_available = False
        raise RuntimeError(
            "CUDA GPU is required for this RealMLP run but is unavailable: "
            f"resolved_device={device} torch={torch_version} "
            f"cuda_available={cuda_available}."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_file = args.fold_file or args.output_dir / "canonical_folds.npz"

    train_path = resolve_input_path(
        args.train,
        filename="train.csv",
        source_hint="playground-series-s6e2",
    )
    test_path = resolve_input_path(
        args.test,
        filename="test.csv",
        source_hint="playground-series-s6e2",
    )
    if args.original_stats and args.original is not None:
        args.original = resolve_input_path(
            args.original,
            filename="Heart_Disease_Prediction.csv",
            source_hint="s6e4-original-dataset",
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
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
        original_source_rows = len(original)
    else:
        X = train[base_features].copy()
        X_test = test[base_features].copy()
        original_stat_features = []
        original_source_rows = 0
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

    print(
        f"rows={len(train):,} test={len(test):,} features={X.shape[1]} "
        f"categorical={len(categorical_features)} original_stats={len(original_stat_features)}"
    )
    print(
        f"device={device} n_cv={args.n_cv} "
        f"n_ens={args.n_ens} epochs={args.n_epochs} batch={args.batch_size}"
    )

    for fold in folds_to_run:
        checkpoint = checkpoints / f"fold_{fold}.npz"
        if args.resume and checkpoint.exists():
            print(f"fold={fold} checkpoint_exists=1 action=skip")
            continue

        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        params = realmlp_parameters(args, fold)
        params["tmp_folder"] = args.output_dir / "tmp" / f"fold_{fold}"
        print(f"fold={fold} train={len(train_idx):,} valid={len(valid_idx):,}")
        start = time.time()

        RealMLP_TD_Classifier = realmlp_classifier_class()
        model = RealMLP_TD_Classifier(**params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[valid_idx],
            y[valid_idx],
            cat_col_names=categorical_features,
        )
        valid_prediction = probability_vector(model.predict_proba(X.iloc[valid_idx]))
        test_prediction = probability_vector(model.predict_proba(X_test))
        fold_auc = float(roc_auc_score(y[valid_idx], valid_prediction))
        np.savez_compressed(
            checkpoint,
            valid_idx=valid_idx,
            valid_prediction=valid_prediction,
            test_prediction=test_prediction,
            fold_auc=np.asarray(fold_auc),
            elapsed_seconds=np.asarray(time.time() - start),
        )
        print(
            f"fold={fold} auc={fold_auc:.9f} "
            f"elapsed_seconds={time.time() - start:.1f}"
        )
        del model, valid_prediction, test_prediction
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    missing = [
        fold for fold in range(args.folds) if not (checkpoints / f"fold_{fold}.npz").exists()
    ]
    if missing:
        print(f"run_incomplete=1 missing_folds={missing}")
        return

    oof = np.zeros(len(train), dtype="float64")
    test_predictions: list[np.ndarray] = []
    fold_auc: list[float] = []
    elapsed_seconds: list[float] = []
    for fold in range(args.folds):
        artifact = np.load(checkpoints / f"fold_{fold}.npz")
        valid_idx = artifact["valid_idx"]
        if not np.array_equal(valid_idx, np.flatnonzero(fold_ids == fold)):
            raise ValueError(f"Fold {fold} checkpoint indices do not match protocol")
        oof[valid_idx] = artifact["valid_prediction"]
        test_predictions.append(artifact["test_prediction"])
        fold_auc.append(float(artifact["fold_auc"]))
        elapsed_seconds.append(float(artifact["elapsed_seconds"]))
    test_prediction = np.mean(test_predictions, axis=0)
    overall_auc = float(roc_auc_score(y, oof))

    stem = (
        f"realmlp_{args.representation}"
        if args.original_stats
        else f"realmlp_raw_{args.representation}"
    )
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
        "model": "RealMLP_TD_Classifier",
        "representation": args.representation,
        "feature_set": args.feature_set,
        "overall_oof_auc": overall_auc,
        "fold_auc": fold_auc,
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc, ddof=1)),
        "elapsed_seconds_by_fold": elapsed_seconds,
        "competition_train_rows": int(len(train)),
        "original_stat_source_rows": int(original_source_rows),
        "feature_count": int(X.shape[1]),
        "categorical_feature_count": len(categorical_features),
        "original_stat_feature_count": len(original_stat_features),
        "engineered_feature_count": len(engineered_features),
        "folds": args.folds,
        "seed": args.seed,
        "n_cv": args.n_cv,
        "n_ens": args.n_ens,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "device": resolve_device(args.device),
        "original_stats": args.original_stats,
        "original_smoothing": args.original_smoothing,
        "original_pairs": args.original_pairs,
    }
    (args.output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"artifacts={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
