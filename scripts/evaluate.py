#!/usr/bin/env python3
"""Independently evaluate research-scope DARC predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default="artifacts/data/splits_grouped/test.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="artifacts/metrics/darc_grouped_test.json")
    args = parser.parse_args()
    metrics = evaluate(args.truth, args.predictions, args.output)
    print(f"Samples: {metrics['sample_count']}")
    print(f"Micro-F1: {metrics['category_metrics']['micro_f1']:.6f}")
    print(f"Primary-category accuracy: {metrics['category_metrics']['primary_category_accuracy']:.6f}")
    print(f"Mean description token F1: {metrics['description_metrics']['mean_token_f1']:.6f}")
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
