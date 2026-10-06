# Codex Kaggle Arena Wiki

This wiki documents a human-directed, OpenAI Codex-assisted competitive
machine-learning workflow. The repository's historical GitHub slug still uses
`antigravity-kaggle-arena`, but Antigravity is no longer the active agent.

## Start here

- [[01-Mission-and-Architecture|Mission, evidence loop, and Codex workflow]]
- [[02-Skills-and-Modules|Skills, feature engineering, and core modules]]
- [[03-Evolution-Playbook|V1–V17 evolution playbook]]
- [[Playground-Series-S6E2|Complete S6E2 solution and submission report]]

## Verified benchmark

| Competition | Metric | Cross-fitted OOF | Kaggle Private / Public | Winner benchmark | Status |
|---|---|---:|---:|---:|---|
| [Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2) | ROC-AUC | **`0.955754551`** | **`0.95535` / `0.95394`** | `0.95535` Private | **Winner score matched** |

Submission `56862032` was made after the competition closed. It validates the
technical result but does not alter the official historical ranking.

## What changed

- The project is now documented as **Codex Kaggle Arena**.
- CV, Public, and Private scores are never mixed.
- The full report covers accepted and rejected experiments from V1 through V17.
- Technical decisions, invoked skills, prompting strategy, and Codex token
  counters are disclosed.
- GitHub Wiki deployment now copies versioned documentation instead of
  regenerating obsolete score claims.
