"""Create and validate a PTCG competition bundle without submitting it."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_MEMBERS = {"main.py", "deck.csv"}


def validate_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain exactly 60 card IDs, found {len(deck)}")
    return deck


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_deck(ROOT / "deck.csv")
    with tarfile.open(output, "w:gz") as archive:
        archive.add(ROOT / "agent.py", arcname="main.py")
        archive.add(ROOT / "deck.csv", arcname="deck.csv")
    with tarfile.open(output, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    if names != REQUIRED_MEMBERS:
        raise ValueError(f"invalid archive layout: expected {REQUIRED_MEMBERS}, found {names}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/ptcg_starter.tar.gz"))
    args = parser.parse_args()
    print(build(args.output).resolve())


if __name__ == "__main__":
    main()
