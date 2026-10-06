"""Run a complete local CABT game between two copies of the starter agent."""

from __future__ import annotations

from kaggle_environments import make

from agent import agent


def main() -> None:
    environment = make("cabt", debug=True)
    environment.run([agent, agent])
    statuses = [state.status for state in environment.state]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"starter agent failed local CABT run: {statuses}")
    print({"statuses": statuses, "rewards": [state.reward for state in environment.state]})


if __name__ == "__main__":
    main()
