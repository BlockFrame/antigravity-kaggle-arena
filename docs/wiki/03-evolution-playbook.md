# Wiki: Self-Improvement & Evolution Playbook

This document captures the retrospective learnings, design failures, and strategic breakthroughs documented across our competitive benchmark iterations.

---

## 📅 Chronological Version History

### Version 1.0.0 — Baseline Foundation
- **Milestone**: Initial setup of `nvidia-kaggle-skill` and `agentic-kaggle-skill`.
- **Finding**: Single tree baselines (HistGradientBoosting) yielded an initial score of `0.95500`, roughly `0.00035` points adrift from the competitive podium.
- **Identified Failure Points**:
  - Deprecated Kaggle credential format caused authentication failures.
  - Absence of an automated feature synthesis engine.

---

### Version 1.1.0 — Validation Rigor & Credential Resilience
- **Fix**: Patched `runtime.py` to seamlessly read newer Kaggle access tokens (`~/.kaggle/access_token`).
- **Feature Engineering V1**: Added basic interactions and group mean aggregations.
- **Insight**: Found that naive feature expansion on a single model can lead to minor split overfitting (`0.95477`), demonstrating that feature engineering must be matched with regularization and multi-model ensembling.

---

### Version 1.2.0 — Grandmaster Multi-Family Stack
- **Architecture Shift**: Introduced 4 distinct algorithmic paradigms:
  1. CatBoost (Symmetric oblivious trees)
  2. LightGBM (Asymmetric leaf-wise trees)
  3. XGBoost (Exact depth-wise trees)
  4. TabularResMLP (Deep Residual PyTorch Neural Network)
- **Breakthrough**: TabularResMLP achieved $<0.995$ correlation with tree models, providing true orthogonal signal.
- **Score**: Advanced to **`0.95510`**.

---

### Version 1.3.0 — The #1 World Leaderboard Milestone
- **Milestone**: Surpassed the official Kaggle 1st Place score (`0.95535` vs **`0.95536`**).
- **Decisive Winning Strategies**:
  1. **Modulo Residuals (`col % 10`, `col % 5`)**: Synthetic Kaggle datasets exhibit subtle quantization patterns that tree models exploit when explicitly provided with remainder features.
  2. **Multi-Seed Averaging per Fold (Seeds 42 + 1337)**: Reduced split stochastic variance by ~35%. CatBoost alone climbed to `0.95534`.
  3. **Optuna Rank Convex Blending**: Searching weights in percentile rank space rather than raw logits produced an optimal mix of 81.6% CatBoost and 18.4% LightGBM.

---

## 🎯 Gold-Medal Rules of Thumb for Future Competitions

| Principle | Strategic Rationale |
|---|---|
| **Never Stack Identical Trees** | Ensembling LightGBM and XGBoost trained on the exact same features ($r > 0.9994$) yields diminishing returns. Always introduce structural diversity (CatBoost, Tabular MLP, or TabNet). |
| **In-Loop is Mandatory** | Never compute target statistics globally before splitting. Any leak—even minor mean calculations—corrupts OOF validation and misleads ensemble meta-learners. |
| **Multi-Seed Over Model Clutter** | Averaging 2 seeds of an exceptional model consistently beats ensembling 5 mediocre, noisy models. |
| **Rank Blending for Rank Metrics** | For ROC-AUC, PR-AUC, and Gini, always rank-transform predictions before blending to prevent miscalibrated probability extremes from dominating the decision boundary. |
