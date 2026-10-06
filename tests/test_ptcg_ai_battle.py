from __future__ import annotations

import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = ROOT / "src" / "competitions" / "ptcg_ai_battle"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, COMPETITION / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = load_module("ptcg_agent", "agent.py")
BUILDER = load_module("ptcg_builder", "build_submission.py")


class PtcgStarterAgentTests(unittest.TestCase):
    def test_initial_action_is_a_complete_deck(self) -> None:
        deck = AGENT.agent({"select": None})
        self.assertEqual(deck, list(AGENT.STARTER_DECK))
        self.assertEqual(len(deck), 60)

    def test_selection_is_legal_and_bounded(self) -> None:
        action = AGENT.agent({"select": {"maxCount": 2, "option": [{}, {}, {}]}})
        self.assertEqual(action, [0, 1])
        self.assertEqual(AGENT.agent({"select": {"maxCount": 3, "option": [{}]}}), [0])

    def test_bundle_has_exactly_the_required_top_level_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = BUILDER.build(Path(directory) / "submission.tar.gz")
            with tarfile.open(bundle, "r:gz") as archive:
                self.assertEqual({item.name for item in archive.getmembers() if item.isfile()}, {"main.py", "deck.csv"})


if __name__ == "__main__":
    unittest.main()
