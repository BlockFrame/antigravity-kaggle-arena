"""Evaluate the starter agent locally with seats alternated.

This intentionally reports match outcomes rather than pretending a small
sample is a stable ladder rating.  It is the common gate for every policy or
deck change before a Kaggle submission is considered.
"""

from __future__ import annotations

import argparse
from collections import Counter

from kaggle_environments import make

from agent import agent


def play(starter_first: bool, opponent: str = "random") -> int:
    environment = make("cabt", debug=True)
    agents = [agent, opponent] if starter_first else [opponent, agent]
    environment.run(agents)
    starter_index = 0 if starter_first else 1
    reward = environment.state[starter_index].reward
    if reward not in {-1, 0, 1}:
        raise RuntimeError(f"invalid local evaluation result: {reward!r}")
    return int(reward)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", default="random")
    args = parser.parse_args()
    if args.games < 2:
        raise ValueError("use at least two games so both seats can be sampled")

    results = Counter(play(game % 2 == 0, args.opponent) for game in range(args.games))
    points = results[1] + 0.5 * results[0]
    print(
        {
            "games": args.games,
            "opponent": args.opponent,
            "wins": results[1],
            "draws": results[0],
            "losses": results[-1],
            "score_rate": points / args.games,
        }
    )


if __name__ == "__main__":
    main()
