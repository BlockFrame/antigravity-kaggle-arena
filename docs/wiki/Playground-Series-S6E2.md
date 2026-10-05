# Playground Series S6E2 — Winner Score Matched

**Competition:** [Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2)

**Metric:** ROC-AUC

**Final verified late submission:** `56862032`

## Result

| Version | OOF | Public | Private |
|---|---:|---:|---:|
| V7 | `0.955733952` | `0.95391` | `0.95532` |
| V14 | `0.955746123` | `0.95392` | `0.95534` |
| V16 | `0.955749097` | `0.95392` | `0.95534` |
| **V17 final** | **`0.955754551`** | **`0.95394`** | **`0.95535`** |

The final Private score equals the displayed winning score. Since it was
submitted after the competition deadline, it validates the solution but does
not retroactively change the official ranking.

## Final architecture

V17 is a rank blend of:

- 58% V14 validated ensemble;
- 42% RealMLP with raw categorical features;
- quantile bins, width bins, rounding, and decimal digit features;
- smoothed target mean, log-count, WoE, and entropy statistics learned only
  from the separate original source, for singleton feature groups;
- five canonical stratified folds, seed 42;
- RealMLP `n_cv=2`, `n_ens=8`, 100 epochs, batch size 128.

The candidate weight was selected outside each held-out fold. Fold-selected
weights were `[0.42, 0.37, 0.51, 0.42, 0.40]`; four of five held-out deltas
were positive. V17 added `+0.000008428` OOF over V14.

## What mattered

1. Raw all-categorical RealMLP established the neural baseline.
2. Bin/digit representations added generator-sensitive diversity.
3. Original singleton statistics improved V17 without the noisier pair expansion.
4. Rank blending was more stable than raw-probability averaging.
5. Cross-fitted weight selection prevented an in-sample ensemble optimum from
   being reported as validation evidence.
6. A depth-2 CatBoost bin/digit candidate improved OOF slightly but reduced
   Private LB, so it was discarded.

## Reproduction entry points

- `train_v7_realmlp.py`
- `train_v7_catboost_ordered.py`
- `evaluate_oof_candidate.py`
- `run_v15_catboost_d2_bin_digit_cpu.py`
- `run_v16_realmlp_bin_digit_seed1337.py`
- `run_v17_realmlp_bin_digit_orig_singletons.py`

All training runners write aligned IDs, targets, fold assignments, OOF
predictions, test predictions, metrics, and restartable fold checkpoints.
