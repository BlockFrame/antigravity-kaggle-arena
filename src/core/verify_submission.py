#!/usr/bin/env python3
"""Submission Verification Gate.

Verifies:
1. File exists and is non-empty.
2. Exact matching row count against sample_submission or test set.
3. Correct column names and order.
4. No NaN, null, or infinite values.
5. Reasonable value ranges (e.g. valid probabilities [0, 1] for classification).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def verify_submission(
    submission_path: str | Path,
    sample_path: str | Path | None = None,
    expected_rows: int | None = None,
    is_probability: bool = True,
) -> bool:
    sub_p = Path(submission_path)
    if not sub_p.exists():
        print(f"❌ Error: Submission file '{sub_p}' does not exist.", file=sys.stderr)
        return False

    try:
        df_sub = pd.read_csv(sub_p)
    except Exception as e:
        print(f"❌ Error reading submission file: {e}", file=sys.stderr)
        return False

    print(f"🔍 Checking submission: {sub_p.name} ({len(df_sub):,} rows, {len(df_sub.columns)} cols)")

    # 1. Null check
    null_count = df_sub.isnull().sum().sum()
    if null_count > 0:
        print(f"❌ FAILED: Found {null_count} null/NaN values in submission!", file=sys.stderr)
        return False

    # 2. Inf check
    num_cols = df_sub.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if np.isinf(df_sub[c]).any():
            print(f"❌ FAILED: Found infinite values in column '{c}'!", file=sys.stderr)
            return False

    # 3. Check against sample submission
    if sample_path:
        sample_p = Path(sample_path)
        if sample_p.exists():
            df_sample = pd.read_csv(sample_p)
            if len(df_sub) != len(df_sample):
                print(f"❌ FAILED: Row count mismatch! Submission: {len(df_sub)}, Expected: {len(df_sample)}", file=sys.stderr)
                return False
            if list(df_sub.columns) != list(df_sample.columns):
                print(f"❌ FAILED: Column mismatch! Submission: {list(df_sub.columns)}, Expected: {list(df_sample.columns)}", file=sys.stderr)
                return False
            if "id" in df_sub.columns and "id" in df_sample.columns:
                if not (df_sub["id"].values == df_sample["id"].values).all():
                    print("❌ FAILED: 'id' column order or values do not match sample_submission!", file=sys.stderr)
                    return False

    # 4. Check expected row count if provided
    if expected_rows and len(df_sub) != expected_rows:
        print(f"❌ FAILED: Expected {expected_rows} rows, found {len(df_sub)}", file=sys.stderr)
        return False

    # 5. Check probability bounds
    if is_probability:
        target_cols = [c for c in df_sub.columns if c.lower() != "id"]
        for c in target_cols:
            if pd.api.types.is_numeric_dtype(df_sub[c]):
                min_val = df_sub[c].min()
                max_val = df_sub[c].max()
                if min_val < -1e-5 or max_val > 1.00001:
                    print(f"❌ FAILED: Values out of bounds for probabilities in '{c}': [{min_val:.4f}, {max_val:.4f}]", file=sys.stderr)
                    return False

    print("✅ PASSED: Submission file is strictly valid and ready for Kaggle scoring!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Kaggle submission integrity")
    parser.add_argument("submission", help="Path to submission.csv")
    parser.add_argument("--sample", help="Path to sample_submission.csv", default=None)
    args = parser.parse_args()

    ok = verify_submission(args.submission, sample_path=args.sample)
    sys.exit(0 if ok else 1)
