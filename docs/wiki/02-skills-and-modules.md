# Skills and core modules

## Invoked Codex skills

The project installed 166 K-Dense Scientific Agent Skills and one NVIDIA Kaggle
skill under `.agents/skills/`. Installation does not imply use: the audited
workflow invoked only the relevant subset.

| Skill | How it influenced the work |
|---|---|
| Codex `skill-installer` | Installed GitHub skill repositories project-scoped and verified manifests |
| NVIDIA [`nvidia-kaggle-skill`](https://github.com/NVIDIA/nvidia-kaggle) | Structured Kaggle discovery, kernel push/poll/download, quota inspection, and guarded submission |
| K-Dense `exploratory-data-analysis` | Guided aggregate-safe dataset inspection and leakage review |
| K-Dense `scikit-learn` | Guided stratified validation, ROC-AUC, and OOF evaluation |
| K-Dense `statistical-analysis` | Guided fold-delta and stability interpretation for very small gains |
| Browser control | Diagnosed the signed-in Kaggle accelerator state when metadata and runtime disagreed |
| `openai-docs` | Used for Codex/Headroom configuration checks, not for model selection |
| `imagegen` | Generated the current repository banner after V17 was complete |

The skills supplied procedures and safety boundaries. They did not replace
measured experiment evidence.

## Competition modules

### Frozen fold construction — `src/core/cv_folds.py`

Creates deterministic stratified fold assignments shared by all model
families. Reusing the same fold IDs permits row-for-row OOF comparison.

### Candidate evaluator — `evaluate_oof_candidate.py`

Checks ID/target/fold alignment, rank-transforms predictions, selects a
candidate weight on non-held-out folds, and scores the held-out fold. This is
the central protection against reporting an in-sample blend optimum.

### RealMLP trainer — `train_v7_realmlp.py`

Supports the representations tested from V7 through V17:

- raw all-categorical;
- hybrid numeric/categorical;
- raw plus bin/digit features;
- optional original-source statistics;
- singleton-only or singleton-plus-pair original groups;
- configurable seed, folds, `n_cv`, `n_ens`, epochs, and GPU requirement.

### CatBoost trainer — `train_v7_catboost_ordered.py`

Provides shallow Ordered/Plain CatBoost candidates on the same canonical folds.
CPU runners made these long jobs independent of the user's local machine.

### Feature engineering — `src/core/feature_engineering_v2.py`

Creates multiple views of continuous variables, including bins, rounding, and
digit/remainder structure. These views were useful because the Playground data
was synthetic and retained discrete generator artifacts.

### In-loop target encoding — `src/core/in_loop_target_encoder.py`

For category `c`, count `n_c`, category target mean `ȳ_c`, global prior `μ`,
and smoothing `m`:

```text
TE(c) = (n_c × ȳ_c + m × μ) / (n_c + m)
```

Mappings are learned on the active training partition and frozen before
validation transformation.

### Rank and ridge ensembling — `src/core/ridge_ensemble.py`

Rank blending was preferred for ROC-AUC because it combines ordering rather
than incompatible probability scales. Ridge/logistic stacking remains
available but was not the final V17 combiner.

### Submission gate — `src/core/verify_submission.py`

Checks:

- exact row count;
- ID and column alignment;
- absence of NaN/Inf;
- one prediction per test row;
- valid numeric bounds.

Training and submission are intentionally separate operations.

## Project-scoped skill provenance

- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)
- [NVIDIA/nvidia-kaggle](https://github.com/NVIDIA/nvidia-kaggle)

K-Dense method citation:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. <https://doi.org/10.48550/arXiv.2609.00065>
