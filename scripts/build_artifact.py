#!/usr/bin/env python3
"""Build an integrity-checked DARC retrieval artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import build_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("research", "competition"))
    parser.add_argument("--data-artifacts", default="artifacts/data")
    parser.add_argument("--features", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    data = Path(args.data_artifacts)
    if args.scope == "research":
        feature_root = Path(args.features or "artifacts/features/research")
        split = data / "splits_grouped/train.json"
        summary = data / "splits_grouped/split_summary.json"
        output = args.output or "artifacts/runtime/research"
        feature_file = feature_root / "train.npz"
    else:
        feature_root = Path(args.features or "artifacts/features/competition")
        split = data / "splits_competition/all_labeled.json"
        summary = None
        output = args.output or "artifacts/runtime/competition"
        feature_file = feature_root / "all_labeled.npz"
    result = build_artifact(
        features_path=feature_file,
        split_path=split,
        provenance_path=feature_root / "provenance.json",
        output_dir=output,
        data_scope=args.scope,
        split_summary_path=summary,
    )
    print(f"Wrote DARC {args.scope} artifact to {output}")
    print(f"Artifact fingerprint: {result['artifact_fingerprint']}")


if __name__ == "__main__":
    main()
