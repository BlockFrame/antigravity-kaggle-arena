#!/usr/bin/env python3
"""Advanced Multi-Representation Feature Engineering (V2).

Implements winning strategies from 1st and 2nd place solutions:
1. Multi-scale Binning (qcut and cut for Age, BP, Cholesterol, Max HR).
2. Digit / Modulo residual features (captures synthetic generation artifacts).
3. Risk Ratios and Interaction terms (clinical cardiac indicators).
4. High-cardinality group aggregations (mean, std, min, max, deviation).
5. Frequency encoding across all discrete attributes.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def generate_v2_features(
    df: pd.DataFrame,
    target_col: str | None = "Heart Disease",
) -> pd.DataFrame:
    data = df.copy()

    # Drop target or identifiers from feature transformations
    ignore_cols = {c for c in [target_col, "id", "fold"] if c in data.columns}
    base_cols = [c for c in data.columns if c not in ignore_cols]

    # Clean col names map
    col_map = {c.lower().replace(" ", "_"): c for c in base_cols}

    # 1. DIGIT & MODULO RESIDUAL FEATURES (Critical for synthetic tabular data)
    num_vars = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    for col in num_vars:
        if col in data.columns:
            # Last digit / rounding residuals
            data[f"{col}_mod10"] = (data[col] % 10).astype(np.float32)
            data[f"{col}_mod5"] = (data[col] % 5).astype(np.float32)
            data[f"{col}_is_multiple10"] = ((data[col] % 10) == 0).astype(np.float32)

    # 2. MULTI-SCALE BINNING (Equal width and Quantile cuts)
    if "Age" in data.columns:
        data["Age_bin5"] = (data["Age"] // 5).astype(np.float32)
        data["Age_bin10"] = (data["Age"] // 10).astype(np.float32)
    if "BP" in data.columns:
        data["BP_bin10"] = (data["BP"] // 10).astype(np.float32)
        data["BP_bin20"] = (data["BP"] // 20).astype(np.float32)
    if "Cholesterol" in data.columns:
        data["Chol_bin25"] = (data["Cholesterol"] // 25).astype(np.float32)
        data["Chol_bin50"] = (data["Cholesterol"] // 50).astype(np.float32)
    if "Max HR" in data.columns:
        data["MaxHR_bin10"] = (data["Max HR"] // 10).astype(np.float32)

    # 3. CLINICAL DOMAIN RATIOS & PRODUCTS
    if "Age" in data.columns and "Max HR" in data.columns:
        # Tanaka formula: 208 - 0.7 * Age
        theoretical_max_hr = 208.0 - 0.7 * data["Age"]
        data["hr_reserve_ratio"] = (data["Max HR"] / (theoretical_max_hr + 1e-5)).astype(np.float32)
        data["age_hr_product"] = (data["Age"] * data["Max HR"]).astype(np.float32)
        data["age_hr_diff"] = (data["Age"] - data["Max HR"]).astype(np.float32)

    if "BP" in data.columns and "Cholesterol" in data.columns:
        data["chol_bp_ratio"] = (data["Cholesterol"] / (data["BP"] + 1e-5)).astype(np.float32)
        data["chol_bp_product"] = (data["Cholesterol"] * data["BP"]).astype(np.float32)
        # Framingham-like risk proxy: BP * Age / Cholesterol
        if "Age" in data.columns:
            data["cardio_risk_index"] = ((data["BP"] * data["Age"]) / (data["Cholesterol"] + 1e-5)).astype(np.float32)

    if "ST depression" in data.columns and "Slope of ST" in data.columns:
        data["st_slope_product"] = (data["ST depression"] * data["Slope of ST"]).astype(np.float32)
        data["st_depression_nonzero"] = (data["ST depression"] > 0).astype(np.float32)

    # 4. FREQUENCY ENCODING (Base features + Binned features)
    encode_candidates = [
        "Sex", "Chest pain type", "FBS over 120", "EKG results",
        "Exercise angina", "Slope of ST", "Number of vessels fluro",
        "Thallium", "Age_bin5", "BP_bin10", "Chol_bin25"
    ]
    for col in encode_candidates:
        if col in data.columns:
            freq = data[col].value_counts(normalize=True)
            data[f"{col}_freq"] = data[col].map(freq).astype(np.float32)

    # 5. HIGH-LEVERAGE GROUP AGGREGATIONS
    group_keys = ["Chest pain type", "Sex", "Thallium", "Number of vessels fluro"]
    metrics_to_agg = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
    
    for grp in group_keys:
        if grp not in data.columns:
            continue
        for met in metrics_to_agg:
            if met not in data.columns:
                continue
            grouped = data.groupby(grp)[met]
            mean_val = grouped.transform("mean").astype(np.float32)
            std_val = grouped.transform("std").fillna(0).astype(np.float32)
            data[f"{met}_mean_by_{grp}"] = mean_val
            data[f"{met}_diff_from_mean_{grp}"] = (data[met] - mean_val).astype(np.float32)
            data[f"{met}_zscore_by_{grp}"] = ((data[met] - mean_val) / (std_val + 1e-5)).astype(np.float32)

    return data


if __name__ == "__main__":
    print("Feature Engineering V2 ready.")
