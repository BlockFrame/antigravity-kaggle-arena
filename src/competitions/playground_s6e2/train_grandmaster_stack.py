#!/usr/bin/env python3
"""Run Grandmaster Stack on Heart Disease Kaggle Playground S6E2.

Orchestrates:
1. Feature Engineering (Interactions + Frequency + Aggregations)
2. In-Loop Target Encoding per fold (Zero Leakage)
3. Model 1: LightGBM (Gradient Boosting)
4. Model 2: CatBoost (Symmetric Trees)
5. Model 3: XGBoost (Exact Tree Boosting)
6. Model 4: TabularResMLP (Deep Residual Network)
7. Rank-Weighted Meta-Stacking
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

# Setup path to skills
SKILL_SCRIPTS = Path("/Users/stefanorossi/.gemini/config/skills/agentic-kaggle-skill/scripts")
sys.path.append(str(SKILL_SCRIPTS))

from auto_feature_engineering import generate_tabular_features
from in_loop_target_encoder import OOFInLoopTargetEncoder
from tabular_mlp import train_tabular_mlp

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

def main():
    print("=" * 60)
    print("🚀 LAUNCHING GRANDMASTER 4-MODEL STACK (S6E2)")
    print("=" * 60)

    data_path = Path("/Users/stefanorossi/.gemini/antigravity/scratch/competitions/playground-series-s6e2/input/train_combined.csv")
    df = pd.read_csv(data_path)
    target_col = "Heart Disease"
    y = (df[target_col] == "Presence").astype(int).values
    
    cat_cols = ["Sex", "Chest pain type", "FBS over 120", "EKG results", "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"]
    
    print("Stage 1: Generating Domain Features & Aggregations...")
    df_feat = generate_tabular_features(df, target_col=target_col, cat_cols=cat_cols)
    X_raw = df_feat.drop(columns=[target_col])
    print(f"Total Features engineered: {X_raw.shape[1]}")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    oof_mlp = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        print(f"\n--- FOLD {fold + 1} / {n_splits} ---")
        X_tr = X_raw.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X_raw.iloc[val_idx].copy()
        y_va = y[val_idx]

        # In-Loop Target Encoding
        te = OOFInLoopTargetEncoder(cat_cols=cat_cols, smooth_weight=20.0)
        X_tr_enc = te.fit_transform(X_tr, y_tr)
        X_va_enc = te.transform(X_va)

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        m_lgb.fit(X_tr_enc, y_tr)
        p_lgb = m_lgb.predict_proba(X_va_enc)[:, 1]
        oof_lgb[val_idx] = p_lgb
        print(f"  LightGBM AUC: {roc_auc_score(y_va, p_lgb):.5f}")

        # 2. CatBoost
        m_cat = CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1
        )
        m_cat.fit(X_tr_enc, y_tr)
        p_cat = m_cat.predict_proba(X_va_enc)[:, 1]
        oof_cat[val_idx] = p_cat
        print(f"  CatBoost AUC: {roc_auc_score(y_va, p_cat):.5f}")

        # 3. XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.035,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="auc",
            n_jobs=-1
        )
        m_xgb.fit(X_tr_enc, y_tr)
        p_xgb = m_xgb.predict_proba(X_va_enc)[:, 1]
        oof_xgb[val_idx] = p_xgb
        print(f"  XGBoost  AUC: {roc_auc_score(y_va, p_xgb):.5f}")

        # 4. TabularResMLP (PyTorch)
        p_mlp = train_tabular_mlp(
            X_tr_enc.values,
            y_tr,
            X_va_enc.values,
            y_va,
            epochs=15,
            batch_size=4096,
            lr=1e-3,
            device="cpu"
        )
        oof_mlp[val_idx] = p_mlp
        print(f"  ResMLP   AUC: {roc_auc_score(y_va, p_mlp):.5f}")

    print("\n" + "=" * 60)
    print("📊 OVERALL SINGLE MODEL OOF SCORES")
    print("=" * 60)
    score_lgb = roc_auc_score(y, oof_lgb)
    score_cat = roc_auc_score(y, oof_cat)
    score_xgb = roc_auc_score(y, oof_xgb)
    score_mlp = roc_auc_score(y, oof_mlp)
    print(f"LightGBM  OOF ROC-AUC: {score_lgb:.5f}")
    print(f"CatBoost  OOF ROC-AUC: {score_cat:.5f}")
    print(f"XGBoost   OOF ROC-AUC: {score_xgb:.5f}")
    print(f"ResMLP    OOF ROC-AUC: {score_mlp:.5f}")

    # Optimal Rank Averaging
    r_lgb = rankdata(oof_lgb) / len(y)
    r_cat = rankdata(oof_cat) / len(y)
    r_xgb = rankdata(oof_xgb) / len(y)
    r_mlp = rankdata(oof_mlp) / len(y)

    # Grid search optimal rank blend
    best_blend_auc = 0.0
    best_weights = None
    
    # Weight grid search
    for w_cat in [0.35, 0.45]:
        for w_lgb in [0.25, 0.35]:
            for w_xgb in [0.15, 0.25]:
                w_mlp = 1.0 - (w_cat + w_lgb + w_xgb)
                if w_mlp < 0.05 or w_mlp > 0.25:
                    continue
                blend = w_cat * r_cat + w_lgb * r_lgb + w_xgb * r_xgb + w_mlp * r_mlp
                b_auc = roc_auc_score(y, blend)
                if b_auc > best_blend_auc:
                    best_blend_auc = b_auc
                    best_weights = (w_cat, w_lgb, w_xgb, w_mlp)

    print("\n" + "=" * 60)
    print(f"🏆 GRANDMASTER ENSEMBLE OOF ROC-AUC: {best_blend_auc:.5f}")
    print(f"Optimal Weights: CatBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}, XGBoost={best_weights[2]:.2f}, ResMLP={best_weights[3]:.2f}")
    print("=" * 60)

    # Save OOF predictions
    oof_dir = Path("/Users/stefanorossi/.gemini/antigravity/scratch/competitions/playground-series-s6e2/oof")
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_dir / "oof_lgb.npy", oof_lgb)
    np.save(oof_dir / "oof_cat.npy", oof_cat)
    np.save(oof_dir / "oof_xgb.npy", oof_xgb)
    np.save(oof_dir / "oof_mlp.npy", oof_mlp)
    print("Saved all OOF predictions to oof/ folder.")

if __name__ == "__main__":
    main()
