#!/usr/bin/env python3
"""Ensemble and Blending with Ridge Meta-Learner on Out-Of-Fold Predictions.

Combines predictions from diverse model families (e.g. HistGradientBoosting, CatBoost, RandomForest/Logistic)
using L2-regularized Ridge Regression or Rank Averaging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score


def ridge_oof_stacking(
    oof_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    test_preds_dict: dict[str, np.ndarray] | None = None,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None, float, dict[str, float]]:
    """Stack multiple model predictions using Ridge/Logistic meta-learner.

    Args:
        oof_dict: {model_name: 1D array of OOF probabilities}
        y_true: Ground truth binary target (0 or 1)
        test_preds_dict: Optional {model_name: 1D array of test probabilities}
        alpha: Regularization strength

    Returns:
        (stacked_oof, stacked_test_preds, stacked_auc, model_weights)
    """
    model_names = list(oof_dict.keys())
    X_meta = np.column_stack([oof_dict[m] for m in model_names])

    # Fit Logistic / Ridge meta-model (ensuring strictly valid probability outputs)
    meta_model = LogisticRegression(C=1.0 / alpha, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    meta_model.fit(X_meta, y_true)

    stacked_oof = meta_model.predict_proba(X_meta)[:, 1]
    stacked_auc = float(roc_auc_score(y_true, stacked_oof))

    weights = {m: float(w) for m, w in zip(model_names, meta_model.coef_[0])}

    stacked_test = None
    if test_preds_dict is not None and all(m in test_preds_dict for m in model_names):
        X_test_meta = np.column_stack([test_preds_dict[m] for m in model_names])
        stacked_test = meta_model.predict_proba(X_test_meta)[:, 1]

    return stacked_oof, stacked_test, stacked_auc, weights


def rank_average_ensemble(
    preds_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Combine predictions via percentile rank averaging for robust metric alignment (ROC-AUC)."""
    from scipy.stats import rankdata

    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)

    ranked_preds = np.zeros(len(preds_list[0]))
    for pred, w in zip(preds_list, weights):
        ranked = rankdata(pred) / len(pred)
        ranked_preds += w * ranked

    return ranked_preds
