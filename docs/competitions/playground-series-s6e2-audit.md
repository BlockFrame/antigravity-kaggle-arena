# Playground Series S6E2 — Reproducibility Audit and V6 Plan

## Final outcome update — 2026-10-05

The validation-first sequence described below culminated in V17. Kaggle
submission `56862032` is complete and reports **`0.95394` public / `0.95535`
private**, matching the displayed winning Private score. The final
cross-fitted OOF is `0.955754551`.

The result is a late submission: it verifies model quality but does not alter
the competition's historical official ranking. The final improvement came from
combining the V14 rank blend (58%) with a RealMLP representation containing
raw categorical features, unsupervised bin/digit features, and original-source
singleton statistics (42%).

Audit date: 2026-09-28. Metric: ROC-AUC; higher is better.

## Verified position

The best submitted arena artifact found through the Kaggle API is V5
submission `56651832`: **0.95361 public / 0.95507 private**. The winning
write-up reports **0.95396 public / 0.95535 private** for the selected final
submission, so the current private gap is about **0.00028**. Public scores are
not interchangeable with private scores and are reported separately here.

The original V5 documentation quoted an OOF score over the 630,303-row
combined dataset. This audit evaluates OOF on the 630,000 competition rows
only; the 303 original rows are training anchors, not validation examples.

| Artifact | Synthetic-only OOF AUC | Notes |
|---|---:|---|
| V5 reconstructed rank blend | 0.955430235 | Cat 61.5%, LGB 20.4%, XGB 18.1% |
| V6 OHE logistic | 0.955447125 | All features categorical; original 303 added inside each training fold |
| **V6 fixed family rank blend** | **0.955615645** | 50% OHE logistic, 50% V5 family blend |

The V6 gain over reconstructed V5 is **+0.000185410 OOF AUC**. The fixed blend
improved all five diagnostic folds. Its logistic/V5 Spearman correlation is
`0.996622`: still high, but materially more diverse than adding more seeds of
the same GBDT family.

This is a local result, not a Kaggle score. The generated submission has been
checked for exact IDs, row count, columns, finite values, and range, but has
**not** been submitted.

## Why the previous approach plateaued

The saved GBDT predictions are extremely correlated, and additional seeds
mostly reduce variance without introducing new ranking information. The
winning solutions instead generated genuinely different representations and
model families, then selected a small stable subset.

The strongest evidence from the competition write-ups is:

- The [1st-place solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)
  generated roughly 150 OOF vectors across GBDTs, RealMLP, AutoGluon, RGF,
  TabICL, multiple feature representations, and original-data statistics. It
  used subset selection plus Ridge and warned that CV above roughly 0.95578
  stopped translating reliably to leaderboard gains.
- The [2nd-place solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/2nd-place-solution-avoid-leaks-and-overfitting)
  compared target statistics inside and outside CV, selected low-correlation
  multi-seed candidates, and combined CatBoost with RealMLP predictions.
- The [4th-place solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution)
  found the signal close to linear, reported a strong all-categorical OHE
  logistic baseline, used depth-2 stumps and neural models, and preferred rank
  ensembling over raw-probability averaging.
- The [20th-place solution](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/20th-place-solution-private-0-95533-ridge-stac)
  used an OOF-first workflow, rank-correlation checks, Ridge stacking, and a
  weighted rank average rather than blind submission blending.

## Reproduce V6 locally

Train the diversity model:

```bash
python src/competitions/playground_s6e2/train_v6_ohe_logistic.py \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --combined /path/to/train_combined.csv \
  --output-dir /path/to/v6_ohe_logistic \
  --folds 5 --seed 42 --c 3.0
```

Build the conservative fixed-weight blend from the legacy V5 arrays:

```bash
python src/competitions/playground_s6e2/build_v6_rank_blend.py \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --combined /path/to/train_combined.csv \
  --logistic-oof /path/to/v6_ohe_logistic/ohe_logistic_oof.npz \
  --logistic-test /path/to/v6_ohe_logistic/ohe_logistic_test.npy \
  --v5-oof-dir /path/to/legacy/oof \
  --output-dir /path/to/v6_rank_blend \
  --logistic-weight 0.5
```

The blend script verifies that the generated rows in `train_combined.csv` are
value-aligned with `train.csv` before reusing legacy arrays. It refuses to
silently slice or reorder predictions.

## Next experiments, in priority order

1. Add a true RealMLP portfolio with periodic embeddings and aligned fold IDs.
   This is the clearest missing family in both the top solutions and the local
   correlation matrix. The V7 runner is now available at
   `src/competitions/playground_s6e2/train_v7_realmlp.py` and checkpoints each
   completed outer fold.
2. Add CatBoost `Ordered` and depth-2 stump variants on raw/all-categorical
   representations. Save every OOF/test pair with IDs and fold metadata.
3. Add original-data-only target statistics computed inside each outer fold;
   compare augmentation versus statistics rather than mixing them blindly.
4. Replace the existing in-sample meta score with cross-fitted Ridge using the
   exact same folds as the base OOF generation. Never report a Ridge score
   obtained by fitting and evaluating on the same OOF matrix.
5. Use repeated seeds or a second fixed split to estimate whether a gain of
   less than `0.00005` is stable. Do not select on one split alone.

### V7 RealMLP GPU run

The Kaggle bootstrap prefers the checksum-verified PyTabKit 1.7.3 wheel from
the attached public dataset, so it does not depend on competition-runtime
internet access. Kaggle runs also fail fast if the backend provisions a
CPU-only PyTorch image despite a requested GPU.

The runner defaults to the published high-performing configuration: 100
epochs, batch size 128, `n_cv=2`, `n_ens=8`, Mish, PLR embeddings and
`1-auc_ovr` early stopping. On Kaggle it auto-discovers both current and legacy
input mount layouts and installs the pinned `pytabkit` runtime only when it is
missing.

```bash
python src/competitions/playground_s6e2/train_v7_realmlp.py \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --original /path/to/Heart_Disease_Prediction.csv \
  --output-dir artifacts/v7_realmlp \
  --fold-file artifacts/canonical_folds_seed42.npz
```

Use `--folds-to-run 0,1` to split a long GPU experiment across jobs. Reusing
the same output directory with `--resume` skips completed checkpoints. No
competition submission is performed by this script.

When the run completes, evaluate its marginal contribution without fitting a
weight on the same rows used to report it:

```bash
python src/competitions/playground_s6e2/evaluate_oof_candidate.py \
  --base-oof /path/to/v6_rank_blend_oof.npz \
  --candidate-oof /path/to/realmlp_all_categorical_oof.npz \
  --grid-step 0.05 \
  --output-json /path/to/realmlp_vs_v6.json
```

The evaluator chooses the candidate weight on four folds and scores it on the
held-out fifth fold, repeating this for every fold. Its full-OOF optimum is
labelled diagnostic-only and is never treated as the honest blend score.

### V7 CatBoost Ordered run

The aligned CatBoost runner uses the same canonical outer folds, keeps the 13
raw columns categorical in the all-categorical representation, and appends the
68 external-original statistics as numeric features. Each completed fold is a
restartable checkpoint.

The fold-0 gate justified completing the run: standalone CatBoost reached
`0.956006581` versus `0.956020169` for V6, but their Spearman correlation was
`0.998827950`. A diagnostic rank blend at 45% CatBoost reached `0.956093066`,
or `+0.000072897` over V6 on that fold. This single-fold weight is evidence of
complementarity, not a final or cross-fitted blend estimate.

```bash
python src/competitions/playground_s6e2/train_v7_catboost_ordered.py \
  --train /path/to/train.csv \
  --test /path/to/test.csv \
  --combined /path/to/train_combined.csv \
  --output-dir artifacts/v7_catboost_ordered_d3_allcat \
  --fold-file artifacts/canonical_folds_seed42.npz \
  --boosting-type Ordered \
  --depth 3 \
  --representation all_categorical
```

No submission or upload is performed by either V7 trainer.

### V7 three-way rank blend

The final staged cross-fit first fixes the previously validated V6/RealMLP
rank blend at 30/70, then selects the CatBoost weight on four folds and scores
it on the held-out fifth fold. Selected CatBoost weights were
`[0.25, 0.30, 0.20, 0.25, 0.25]`; the median deployment weights are 22.5% V6,
52.5% RealMLP, and 25% CatBoost.

The honest cross-fitted OOF AUC is `0.955733952`, a gain of `+0.000013855`
over the V6/RealMLP base and `+0.000118307` over V6. Four held-out folds
improve; fold 1 changes by only `-0.000002102`. The local submission is built
by `src/competitions/playground_s6e2/build_v7_three_way_rank_blend.py`; it is
not uploaded automatically. With explicit user approval it was submitted as
Kaggle reference `56685579`, scoring `0.95391` public and `0.95532` private.
That is `0.00003` below the winner's `0.95535` private benchmark.

## Method provenance

The leakage, split-boundary, and aggregate-only EDA checks followed the
Scientific Agent Skills procedures:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: Library Procedural Knowledge Research Agents*.
arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
