#!/usr/bin/env python3
"""Bootstrap the pinned V16 bin/digit RealMLP seed-1337 run on Kaggle."""

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
WORKSPACE = Path("/kaggle/working/s6e2_v16_runner")
FILES = {
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
    output_dir = Path("/kaggle/working/v16_realmlp_bin_digit_seed1337")
    script = WORKSPACE / "src/competitions/playground_s6e2/train_v7_realmlp.py"
    command = [
        sys.executable,
        str(script),
        "--train",
        str(train),
        "--test",
        str(test),
        "--output-dir",
        str(output_dir),
        "--no-original-stats",
        "--feature-set",
        "bin_digit",
        "--representation",
        "all_categorical",
        "--seed",
        "1337",
        "--batch-size",
        "128",
        "--n-cv",
        "2",
        "--n-ens",
        "8",
        "--n-epochs",
        "100",
        "--require-gpu",
    ]
    print("command=" + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=WORKSPACE)


if __name__ == "__main__":
    main()
