#!/usr/bin/env python3
"""Export frozen CLIP features for research or final competition use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import export_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("research", "competition"))
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--data-artifacts", default="artifacts/data")
    parser.add_argument("--model", required=True, help="Local CLIP ViT-B/32 directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--confirm-parameters-frozen",
        action="store_true",
        help="Required for competition mode before all 1200 labels may enter the index.",
    )
    args = parser.parse_args()
    if args.mode == "competition" and not args.confirm_parameters_frozen:
        parser.error("competition mode requires --confirm-parameters-frozen")
    data_root = Path(args.data_artifacts)
    if args.mode == "research":
        split_paths = {
            name: data_root / f"splits_grouped/{name}.json"
            for name in ("train", "val", "test")
        }
        output = args.output or "artifacts/features/research"
    else:
        split_paths = {"all_labeled": data_root / "splits_competition/all_labeled.json"}
        output = args.output or "artifacts/features/competition"
    export_features(
        mode=args.mode,
        split_paths=split_paths,
        taxonomy_path=data_root / "label_taxonomy.json",
        dataset_dir=args.dataset,
        model_dir=args.model,
        output_dir=output,
        device=args.device,
        batch_size=args.batch_size,
        parameters_frozen=args.confirm_parameters_frozen,
    )
    print(f"Wrote {args.mode} features and provenance to {output}")


if __name__ == "__main__":
    main()
