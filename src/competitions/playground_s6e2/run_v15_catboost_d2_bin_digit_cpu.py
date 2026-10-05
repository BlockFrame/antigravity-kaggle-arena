#!/usr/bin/env python3
"""Bootstrap the pinned V15 CatBoost depth-2 bin/digit CPU run on Kaggle."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path


COMMIT = "fa9a577"
RAW_ROOT = (
    "https://raw.githubusercontent.com/BlockFrame/antigravity-kaggle-arena/"
    f"{COMMIT}"
)
WORKSPACE = Path("/kaggle/working/s6e2_v15_runner")
FILES = {
    "src/competitions/playground_s6e2/train_v7_catboost_ordered.py": (
        "src/competitions/playground_s6e2/train_v7_catboost_ordered.py"
    ),
    "src/competitions/playground_s6e2/train_v7_realmlp.py": (
        "src/competitions/playground_s6e2/train_v7_realmlp.py"
    ),
    "src/core/cv_folds.py": "src/core/cv_folds.py",
}


def resolve_input(filename: str, source_hint: str) -> Path:
    candidates = sorted(
        path
        for path in Path("/kaggle/input").rglob(filename)
        if source_hint in str(path)
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {filename!r} under {source_hint!r}; got {candidates}"
        )
    return candidates[0]


def main() -> None:
    for relative_path, repository_path in FILES.items():
        destination = WORKSPACE / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = f"{RAW_ROOT}/{repository_path}"
        print(f"download={source}", flush=True)
        urllib.request.urlretrieve(source, destination)

    train = resolve_input("train.csv", "playground-series-s6e2")
    test = resolve_input("test.csv", "playground-series-s6e2")
    original = resolve_input("Heart_Disease_Prediction.csv", "s6e4-original-dataset")
    output_dir = Path("/kaggle/working/v15_catboost_d2_bin_digit_cpu")
    script = WORKSPACE / "src/competitions/playground_s6e2/train_v7_catboost_ordered.py"

    command = [
        sys.executable,
        str(script),
        "--train",
        str(train),
        "--test",
        str(test),
        "--original",
        str(original),
        "--output-dir",
        str(output_dir),
        "--no-original-stats",
        "--feature-set",
        "bin_digit",
        "--representation",
        "all_categorical",
        "--boosting-type",
        "Plain",
        "--depth",
        "2",
        "--iterations",
        "5000",
        "--learning-rate",
        "0.02",
        "--task-type",
        "CPU",
    ]
    print("command=" + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=WORKSPACE)


if __name__ == "__main__":
    main()
