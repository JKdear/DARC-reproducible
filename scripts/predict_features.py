#!/usr/bin/env python3
"""Predict from frozen unlabeled CLIP features without reading query labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import write_json
from src.features import load_features
from src.runtime import load_artifact, predict_from_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default=None)
    args = parser.parse_args()
    config, index = load_artifact(args.artifact, args.fingerprint)
    features = load_features(args.features, require_labels=False)
    if features["categories"].astype(str).tolist() != config["categories"]:
        raise SystemExit("query feature taxonomy differs from runtime taxonomy")
    predictions, evidence = predict_from_embeddings(
        features["embeddings"], features["image_ids"].astype(str).tolist(), config, index
    )
    write_json(args.output, predictions)
    evidence_path = args.evidence_output or str(Path(args.output).with_suffix(".evidence.json"))
    write_json(
        evidence_path,
        {
            "artifact_fingerprint": args.fingerprint,
            "data_scope": config["data_scope"],
            "records": evidence,
        },
    )
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Wrote retrieval evidence to {evidence_path}")


if __name__ == "__main__":
    main()
