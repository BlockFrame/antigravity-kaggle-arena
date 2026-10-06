"""Deterministic, dependency-free starter agent for the PTCG CABT simulator.

The competition calls ``agent(observation)``.  At the initial call there is no
selection yet: the returned 60 IDs are interpreted as the deck.  Later calls
must return indices into the engine-provided legal-option array.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# The deck bundled by the official ``kaggle-environments`` CABT environment.
# It is intentionally a reproducible smoke-test deck, not a claim of a strong
# metagame choice.  Deck search becomes a separately measured workstream.
STARTER_DECK: tuple[int, ...] = (
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163,
    1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227,
    1262, 1262,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
)

assert len(STARTER_DECK) == 60


def _value(observation: Any, key: str, default: Any = None) -> Any:
    """Read a field from Kaggle's Struct or a plain test dictionary."""

    if isinstance(observation, Mapping):
        return observation.get(key, default)
    return getattr(observation, key, default)


def _selection_indices(selection: Any) -> list[int]:
    """Return a legal deterministic action for the current selection prompt.

    CABT exposes only legal actions.  Choosing the first ``maxCount`` distinct
    positions is therefore a correct transport baseline and, importantly,
    handles every option schema without hard-coding undocumented option types.
    """

    options: Sequence[Any] = _value(selection, "option", ()) or ()
    max_count = int(_value(selection, "maxCount", 0) or 0)
    return list(range(min(max_count, len(options))))


def agent(observation: Any) -> list[int]:
    """Competition entry point; contains no imports unavailable at submission."""

    selection = _value(observation, "select")
    if selection is None:
        return list(STARTER_DECK)
    return _selection_indices(selection)
