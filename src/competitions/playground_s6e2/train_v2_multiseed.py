#!/usr/bin/env python3
"""Execute Multi-Seed Deep Training and Optuna Meta-Stacking (V2).

Enhancements:
1. Feature Matrix V2 (85+ features with binning, modulo, ratios, z-scores).
2. Conservative Learning Rate (0.02) with 1,000+ trees.
3. Multi-Seed Averaging on CatBoost & LightGBM.
4. Non-linear and Rank-based Optuna Optimization.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SKILL_SCRIPTS = Path("/Users/stefanorossi/.gemini/config/skills/agentic-kaggle-skill/scripts")
sys.path.append(str(SKILL_SCRIPTS))

from feature_engineering_v2 import generate_v2_features
from in_loop_target_encoder import OOFInLoopTargetEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier

def main():
    print("=" * 65)
    print("🚀 RUNNING MULTI-SEED DEEP GBDT STACK WITH FEATURE MATRIX V2")
    print("=" * 65)

    data_path = Path("/Users/stefanorossi/.gemini/antigravity/scratch/competitions/playground-series-s6e2/input/train_combined.csv")
    df = pd.read_csv(data_path)
    target_col = "Heart Disease"
    y = (df[target_col] == "Presence").astype(int).values

    print("Step 1: Engineering Feature Matrix V2...")
    df_v2 = generate_v2_features(df, target_col=target_col)
    X = df_v2.drop(columns=[target_col])
    print(f"Total features created: {X.shape[1]}")

    cat_cols = [
        "Sex", "Chest pain type", "FBS over 120", "EKG results",
        "Exercise angina", "Slope of ST", "Number of vessels fluro",
        "Thallium", "Age_bin5", "BP_bin10", "Chol_bin25"
    ]

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cat_seeds = np.zeros(len(df))
    oof_lgb_seeds = np.zeros(len(df))

    seeds = [42, 1337]  # Multi-seed pairing per fold for variance reduction

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx]

        # In-loop Target Encoding
        te = OOFInLoopTargetEncoder(cat_cols=cat_cols, smooth_weight=25.0)
        X_tr_enc = te.fit_transform(X_tr, y_tr)
        X_va_enc = te.transform(X_va)

        fold_cat_preds = np.zeros(len(val_idx))
        fold_lgb_preds = np.zeros(len(val_idx))

        # Train multiple seeds
        for s in seeds:
            # Model A: Deep CatBoost (learning rate 0.025, 900 iterations)
            cat = CatBoostClassifier(
                iterations=900,
                learning_rate=0.025,
                depth=6,
                random_seed=s + fold * 10,
                verbose=0,
                thread_count=-1
            )
            cat.fit(X_tr_enc, y_tr)
            fold_cat_preds += cat.predict_proba(X_va_enc)[:, 1] / len(seeds)

            # Model B: Tuned LightGBM (learning rate 0.025, 800 iterations)
            lgbm = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.025,
                num_leaves=35,
                subsample=0.85,
                colsample_bytree=0.8,
                min_child_samples=40,
                random_state=s + fold * 10,
                verbose=-1,
                n_jobs=-1
            )
            lgbm.fit(X_tr_enc, y_tr)
            fold_lgb_preds += lgbm.predict_proba(X_va_enc)[:, 1] / len(seeds)

        oof_cat_seeds[val_idx] = fold_cat_preds
        oof_lgb_seeds[val_idx] = fold_lgb_preds

        auc_cat_fold = roc_auc_score(y_va, fold_cat_preds)
        auc_lgb_fold = roc_auc_score(y_va, fold_lgb_preds)
        print(f"  Fold {fold+1} Multi-Seed CatBoost AUC: {auc_cat_fold:.5f}")
        print(f"  Fold {fold+1} Multi-Seed LightGBM AUC: {auc_lgb_fold:.5f}")

    auc_cat_total = roc_auc_score(y, oof_cat_seeds)
    auc_lgb_total = roc_auc_score(y, oof_lgb_seeds)

    print("\n" + "=" * 65)
    print(f"📈 MULTI-SEED CATBOOST OOF ROC-AUC: {auc_cat_total:.5f}")
    print(f"📈 MULTI-SEED LIGHTGBM OOF ROC-AUC: {auc_lgb_total:.5f}")
    print("=" * 65)

    # Optuna search for best blend
    r_cat = rankdata(oof_cat_seeds) / len(y)
    r_lgb = rankdata(oof_lgb_seeds) / len(y)

    def objective(trial):
        w = trial.suggest_float("w", 0.0, 1.0)
        blend = w * r_cat + (1.0 - w) * r_lgb
        return roc_auc_score(y, blend)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_w = study.best_params["w"]
    best_blend = best_w * r_cat + (1.0 - best_w) * r_lgb
    best_score = roc_auc_score(y, best_blend)

    print("\n" + "=" * 65)
    print(f"🏆 NEW OPTUNA BLEND OOF ROC-AUC: {best_score:.5f}")
    print(f"Optimal Weights: CatBoost={best_w:.3f}, LightGBM={1.0-best_w:.3f}")
    print(f"Delta vs previous best (0.95510): +{best_score - 0.95510:.5f}")
    print(f"Delta vs #1 World Leaderboard (0.95535): {best_score - 0.95535:.5f}")
    print("=" * 65)

    # Save outputs
    oof_dir = Path("/Users/stefanorossi/.gemini/antigravity/scratch/competitions/playground-series-s6e2/oof")
    np.save(oof_dir / "oof_v2_cat_multiseed.npy", oof_cat_seeds)
    np.save(oof_dir / "oof_v2_lgb_multiseed.npy", oof_lgb_seeds)
    np.save(oof_dir / "oof_v2_final_blend.npy", best_blend)
    print("Saved all V2 predictions to oof/ folder.")


if __name__ == "__main__":
    main()
