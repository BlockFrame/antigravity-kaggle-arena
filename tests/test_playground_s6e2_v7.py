from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.cv_folds import load_or_create_fold_artifact, make_stratified_fold_ids


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "competitions"
    / "playground_s6e2"
    / "train_v7_realmlp.py"
)
SPEC = importlib.util.spec_from_file_location("train_v7_realmlp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V7)


class CanonicalFoldTests(unittest.TestCase):
    def test_fold_ids_are_deterministic_and_complete(self) -> None:
        target = np.array([0, 1] * 50, dtype=np.int8)
        first = make_stratified_fold_ids(target, n_splits=5, seed=42)
        second = make_stratified_fold_ids(target, n_splits=5, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first.tolist()), set(range(5)))
        self.assertTrue(all(np.sum(first == fold) == 20 for fold in range(5)))

    def test_fold_artifact_rejects_id_misalignment(self) -> None:
        ids = np.arange(20)
        target = np.array([0, 1] * 10, dtype=np.int8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "folds.npz"
            load_or_create_fold_artifact(
                path,
                ids=ids,
                target=target,
                n_splits=2,
                seed=42,
            )
            with self.assertRaisesRegex(ValueError, "ids do not align"):
                load_or_create_fold_artifact(
                    path,
                    ids=ids[::-1],
                    target=target,
                    n_splits=2,
                    seed=42,
                )


class OriginalStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = ["Age", "Sex", "Chest pain type", "Thallium"]
        self.original = pd.DataFrame(
            {
                "Age": [40, 40, 50, 60],
                "Sex": [0, 1, 1, 0],
                "Chest pain type": [1, 1, 2, 2],
                "Thallium": [3, 3, 7, 7],
                V7.TARGET: ["Absence", "Presence", "Presence", "Absence"],
            }
        )
        self.train = self.original[self.features].iloc[:3].copy()
        self.test = pd.DataFrame(
            {
                "Age": [70],
                "Sex": [1],
                "Chest pain type": [4],
                "Thallium": [6],
            }
        )

    def test_original_statistics_are_finite_and_row_aligned(self) -> None:
        train, test, added = V7.add_original_statistics(
            self.train,
            self.test,
            self.original,
            base_features=self.features,
            smoothing=10.0,
            include_pairs=True,
        )
        self.assertEqual(len(train), len(self.train))
        self.assertEqual(len(test), len(self.test))
        self.assertGreater(len(added), 0)
        self.assertTrue(np.isfinite(train[added].to_numpy()).all())
        self.assertTrue(np.isfinite(test[added].to_numpy()).all())

    def test_all_categorical_keeps_statistics_numeric(self) -> None:
        train, test, added = V7.add_original_statistics(
            self.train,
            self.test,
            self.original,
            base_features=self.features,
            smoothing=10.0,
            include_pairs=False,
        )
        train, test, categorical = V7.prepare_representation(
            train,
            test,
            base_features=self.features,
            representation="all_categorical",
            hybrid_cardinality=10,
        )
        self.assertEqual(categorical, self.features)
        self.assertTrue(all(str(train[c].dtype) == "category" for c in categorical))
        self.assertTrue(all(train[c].dtype == np.float32 for c in added))
        self.assertTrue(all(test[c].dtype == np.float32 for c in added))


class BinDigitFeatureTests(unittest.TestCase):
    def test_features_are_deterministic_finite_and_train_fitted(self) -> None:
        train = pd.DataFrame(
            {
                "continuous": np.arange(20, dtype=float),
                "binary": [0, 1] * 10,
            }
        )
        test = pd.DataFrame({"continuous": [-10.0, 50.0], "binary": [0, 1]})

        first_train, first_test, added = V7.add_bin_digit_features(
            train,
            test,
            base_features=["continuous", "binary"],
        )
        second_train, second_test, second_added = V7.add_bin_digit_features(
            train,
            test,
            base_features=["continuous", "binary"],
        )

        self.assertEqual(added, second_added)
        self.assertTrue(added)
        self.assertFalse(any("binary" in column for column in added))
        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_test, second_test)
        self.assertTrue(np.isfinite(first_train[added].to_numpy()).all())
        self.assertTrue(np.isfinite(first_test[added].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
