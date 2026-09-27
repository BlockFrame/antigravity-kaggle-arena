#!/usr/bin/env python3
"""Out-Of-Fold In-Loop Target Encoder for Kaggle Tabular Competitions.

Prevents target leakage by computing smooth target statistics strictly on training folds
and applying them to validation and test folds with m-estimate smoothing.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class OOFInLoopTargetEncoder:
    """Computes target encoding strictly inside the CV loop to avoid leakage."""

    def __init__(self, cat_cols: list[str], smooth_weight: float = 15.0):
        self.cat_cols = cat_cols
        self.smooth_weight = smooth_weight
        self.encoding_maps: dict[str, dict] = {}
        self.global_mean: float = 0.5

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        X = X_train.copy()
        self.global_mean = float(np.mean(y_train))
        self.encoding_maps = {}

        for col in self.cat_cols:
            if col not in X.columns:
                continue
            # Calculate smoothed target mean per category
            stats = pd.DataFrame({"cat": X[col], "target": y_train})
            grouped = stats.groupby("cat")["target"].agg(["count", "mean"])
            
            # Smooth formula: (count * mean + m * global_mean) / (count + m)
            counts = grouped["count"]
            means = grouped["mean"]
            smoothed = (counts * means + self.smooth_weight * self.global_mean) / (counts + self.smooth_weight)
            
            mapping = smoothed.to_dict()
            self.encoding_maps[col] = mapping

            # Map to train
            X[f"{col}_te"] = X[col].map(mapping).fillna(self.global_mean).astype(np.float32)

        return X

    def transform(self, X_val: pd.DataFrame) -> pd.DataFrame:
        X = X_val.copy()
        for col, mapping in self.encoding_maps.items():
            if col in X.columns:
                X[f"{col}_te"] = X[col].map(mapping).fillna(self.global_mean).astype(np.float32)
        return X
