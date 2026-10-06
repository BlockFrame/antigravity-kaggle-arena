# The Pokémon Company — PTCG AI Battle Challenge Playground

## Scope

This is the Arena's active simulation-competition case study. The goal is to
develop an agent that plays best-of-three Pokémon Trading Card Game matches in
the CABT simulator. This is not a conventional train/test prediction task:
the score is an evolving skill rating from matches against other submitted
agents.

## Competition contract

| Item | Constraint |
|---|---|
| Final submission deadline | 8 January 2027, 23:59 UTC |
| Daily submissions | 5 |
| Active entries | Only the latest 2 |
| Evaluation | Rating during ladder play, then final Bradley–Terry tournament |
| Bundle | `.tar.gz` containing top-level `main.py` and `deck.csv` |
| Runtime | 2 vCPU, 12.2 GiB RAM, 11.8 GiB disk |
| Submission limit | 197.7 MiB |

The only legal actions are supplied by the engine. Agents must return indices
from that selection set. The initial action is special: it returns the 60-card
deck. Scores are stochastic and dependent on opponents, so promotion requires
local paired matches with seats alternated and confidence intervals, not a
single lucky run.

## Initial baseline — 6 October 2026

`src/competitions/ptcg_ai_battle/` contains a dependency-free agent and a
known-valid starter deck supplied by the official CABT environment. It returns
the first legal choices, is deliberately weak, and exists to validate the full
local-to-submission packaging contract before strategy work starts.

No Kaggle submission has been made from this repository for this competition.

## Research sequence

1. Build an instrumented evaluator with paired games and alternate seats.
2. Convert card metadata and legal-option structures into a state/action feature layer.
3. Implement a tactical deterministic policy: setup, energy attachment, attack, retreat, evolution, and prize-race priorities.
4. Benchmark deck variants independently from policy variants.
5. Add bounded look-ahead through CABT search APIs with a strict per-turn time budget.
6. Train a value/policy model by self-play only after the heuristic baseline is measurable.
7. Maintain stable and experimental active agents; submit only after explicit approval.

## Local commands

Use the isolated Python 3.12 environment because the CABT dependency stack is
not currently compatible with the repository's Python 3.14 environment:

```bash
.venv/ptcg/bin/python src/competitions/ptcg_ai_battle/local_smoke_test.py
.venv/ptcg/bin/python src/competitions/ptcg_ai_battle/evaluate_local.py --games 20
.venv/ptcg/bin/python src/competitions/ptcg_ai_battle/build_submission.py
```

The second command creates a local archive only. Uploading it to Kaggle is a
separate, explicitly authorized operation.
