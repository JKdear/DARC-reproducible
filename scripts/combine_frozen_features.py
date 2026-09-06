#!/usr/bin/env python3
"""Combine frozen research features into a 1200-image competition index input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import combine_research_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-features", default="artifacts/features/research")
    parser.add_argument("--all-labeled-split", default="artifacts/data/splits_competition/all_labeled.json")
    parser.add_argument("--output", default="artifacts/features/competition")
    parser.add_argument(
        "--confirm-parameters-frozen",
        action="store_true",
        help="Required before test labels may enter the final 1200-image index.",
    )
    args = parser.parse_args()
    if not args.confirm_parameters_frozen:
        parser.error("--confirm-parameters-frozen is required")
    root = Path(args.research_features)
    result = combine_research_features(
        train_path=root / "train.npz",
        val_path=root / "val.npz",
        test_path=root / "test.npz",
        test_labels_path=root / "test_labels.npz",
        research_provenance_path=root / "provenance.json",
        all_labeled_split_path=args.all_labeled_split,
        output_dir=args.output,
        parameters_frozen=args.confirm_parameters_frozen,
    )
    details = result["artifacts"]["all_labeled"]
    print(f"Wrote {details['sample_count']}-image competition features to {details['path']}")
    print("Policy: do not use this artifact to report the original grouped-test metrics")


if __name__ == "__main__":
    main()
