#!/usr/bin/env python3
"""Build the portable manifest and frozen DARC data splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--output", default="artifacts/data")
    parser.add_argument(
        "--freeze-splits-from",
        default=None,
        help="Optional split directory whose image-ID membership must be retained.",
    )
    args = parser.parse_args()
    result = prepare_dataset(
        args.dataset,
        args.output,
        reference_split_dir=args.freeze_splits_from,
    )
    counts = {name: len(samples) for name, samples in result["splits"].items()}
    print(f"Prepared {result['manifest']['stats']['sample_count']} labeled images")
    print(f"Frozen grouped split: {counts}")
    print(f"Research files: {Path(args.output) / 'splits_grouped'}")
    print(f"Competition-only full split: {Path(args.output) / 'splits_competition/all_labeled.json'}")


if __name__ == "__main__":
    main()
