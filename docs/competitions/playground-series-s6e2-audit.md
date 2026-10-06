# Playground Series S6E2: final reproducibility audit

Audit finalized: 2026-10-06. Metric: ROC-AUC, higher is better.

## Audited claim

The strongest supported claim is:

> Late submission `56862032` scored `0.95394` Public and `0.95535` Private,
> matching the displayed winning Private score. It does not change the closed
> competition's official historical ranking.

The corresponding cross-fitted OOF score is `0.955754551`.

## Evidence sources

| Claim | Source |
|---|---|
| Submission status and scores | Authenticated Kaggle submission history |
| V17 standalone OOF `0.9557391082984937` | `realmlp_all_categorical_bin_digit_metrics.json` |
| V14 base OOF `0.955746123` | Aligned V14 OOF artifact and evaluator output |
| Final cross-fitted OOF `0.9557545510254963` | V14/V17 held-out fold evaluation |
| Deployment weight 42% | Median of held-out selections `[0.42, 0.37, 0.51, 0.42, 0.40]` |
| Final Public/Private `0.95394/0.95535` | Kaggle submission `56862032` |

No Public or Private value is presented as CV, and no CV value is presented as
a leaderboard score.

## Artifact identity

The local final submission has 270,001 CSV lines: one header plus 270,000 test
predictions.

| Artifact | SHA-256 |
|---|---|
| `submission_candidate_rank_blend.csv` | `4f18ab4d3258f353ed2bb306fc0cc35448990a5ab860bc428aca110d93f86028` |
| V17 OOF NPZ | `85eefcf17a79cc6e7704392ec420dec68141bfd830bcb068b49b3ed3e70f5f3c` |

Artifacts and competition data are not committed to Git.

## Validation invariants

Before evaluating a candidate, the pipeline requires:

1. one prediction per competition training row;
2. identical sample IDs between base and candidate;
3. identical targets;
4. identical frozen fold assignments;
5. one-dimensional finite prediction arrays;
6. weight selection on folds different from the fold being scored;
7. test prediction length equal to the test CSV length;
8. valid submission ID, target column, row count, and finite range.

## Final candidate evaluation

The base was the fixed V14 rank blend. V17 predictions were rank-transformed
before combination. For each held-out fold, the evaluator selected a candidate
weight using only the other four folds.

| Held-out fold | Selected V17 weight | Direction on held-out fold |
|---:|---:|---|
| 0 | `0.42` | Positive |
| 1 | `0.37` | Positive |
| 2 | `0.51` | Positive |
| 3 | `0.42` | Positive |
| 4 | `0.40` | Slightly negative |

The 42% median was used for deployment. Cross-fitted OOF improved by
`0.000008428` over V14.

## Negative controls and rejected evidence

- Re-adding a representation already contained in the base correctly selected
  zero or negligible weight during evaluator testing.
- V15 CatBoost raised aggregate local blend OOF to `0.955750482` but reduced
  Kaggle Private from `0.95534` to `0.95533`; it is not part of V17.
- V16 seed averaging reached `0.955749097` OOF but did not change the Private
  score beyond `0.95534`.
- Earlier `0.95536` and `0.95542` values were local/legacy CV figures and are
  not described as verified Kaggle scores.

## Leakage boundary

Competition targets were used only inside each training partition for
supervised transforms. V17's original-source statistics were learned from the
separate original dataset and consisted of singleton target mean, log-count,
WoE, and entropy features. Original pair statistics were disabled.

The external original data may legitimately improve similarity to the hidden
test distribution, but it also creates a dependency that must be disclosed in
any reproduction.

## Compute boundary

- RealMLP: Kaggle NVIDIA T4, with CUDA fail-fast.
- CatBoost candidates: Kaggle CPU.
- Five folds with checkpoints; remote execution continued independently of the
  local Mac after a Kaggle version was running.
- No training script automatically submitted to the competition.

## Known limitations

- The official competition is closed, so the result is a late-submission
  validation rather than an official rank.
- Kaggle scores are rounded for display; equality at five decimals does not
  prove equality of the unrounded hidden-test AUC.
- The final difference over V14 is very small. Fold consistency and the final
  Kaggle result support it, but they do not make the gain universal.
- A single competition cannot establish that the workflow will generalize to
  other tabular tasks.
- Token counters measure runtime processing, including cached repeated context,
  not unique authored content or model-training compute.

## Reproduction references

- [Complete solution report](playground-series-s6e2.md)
- `src/competitions/playground_s6e2/evaluate_oof_candidate.py`
- `src/competitions/playground_s6e2/run_v17_realmlp_bin_digit_orig_singletons.py`
- `src/core/cv_folds.py`
- `src/core/verify_submission.py`
