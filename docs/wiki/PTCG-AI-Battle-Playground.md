# PTCG AI Battle Challenge Playground

## Status

**Active — baseline validated locally, no Kaggle submission yet.**

This case study develops an agent for best-of-three Pokémon Trading Card Game
matches in the CABT simulator. The target is an evolving skill rating from
games against other agents, followed by a final Bradley–Terry tournament—not a
static prediction metric.

## Competition constraints

| Item | Constraint |
|---|---|
| Final submission deadline | 8 January 2027, 23:59 UTC |
| Daily submissions | 5 |
| Active entries | Latest 2 only |
| Bundle | Top-level `main.py` and `deck.csv` in `.tar.gz` |
| Runtime | 2 vCPU, 12.2 GiB RAM, 11.8 GiB disk |
| Submission limit | 197.7 MiB |

## Baseline contract

The starter agent returns the official CABT environment's known-valid deck at
initialization and then chooses the first legal actions. It is intentionally a
transport and packaging baseline, not a competitive strategy. It validates a
local best-of-three game and produces a correctly structured archive; no
external submission has been consumed.

## Development path

1. Evaluate policy changes in paired local games with seats alternated.
2. Build state/action features from card metadata and legal options.
3. Add a tactical policy for setup, energy, attacks, evolution, retreat, and prize races.
4. Search deck and policy separately before combining them.
5. Add bounded CABT look-ahead, then self-play value/policy learning.
6. Keep stable and experimental agents distinct; submission remains human-approved.

Read the [versioned competition report](https://github.com/BlockFrame/antigravity-kaggle-arena/blob/main/docs/competitions/ptcg-ai-battle-playground.md) for local commands and the evolving evidence log.
