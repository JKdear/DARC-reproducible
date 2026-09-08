#!/usr/bin/env python3
"""Run DARC on one image or a non-recursive directory of unseen images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, write_json
from src.runtime import load_artifact, predict_from_embeddings

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input does not exist: {path}")
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError("input directory has no supported images")
    if len({item.name for item in images}) != len(images):
        raise ValueError("input image names must be unique")
    return images


def verify_model(model_dir: str | Path, expected: dict[str, str]) -> None:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError("model must be a local directory")
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"model file is missing: {path}")
        if file_sha256(path) != digest:
            raise ValueError(f"model file hash mismatch: {name}")


def encode_images(paths: list[Path], model_dir: str | Path, device: str, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("prediction requires torch, transformers, Pillow, and numpy") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    model.eval()
    batches = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        try:
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.detach().cpu().float().numpy())
        finally:
            for image in images:
                image.close()
        print(f"Encoded {min(start + len(batch_paths), len(paths))}/{len(paths)} images")
    return np.concatenate(batches, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/result.json")
    parser.add_argument("--evidence-output", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--require-category-overlap",
        action="store_true",
        help="Prefer a description whose donor categories intersect the predicted set. "
        "Off by default; it does not change predicted categories or the frozen algorithm.",
    )
    args = parser.parse_args()
    config, index = load_artifact(args.artifact, args.fingerprint)
    verify_model(args.model, config["local_model_file_sha256"])
    paths = discover_images(args.input)
    embeddings = encode_images(paths, args.model, args.device, args.batch_size)
    predictions, evidence = predict_from_embeddings(
        embeddings,
        [path.name for path in paths],
        config,
        index,
        require_category_overlap=args.require_category_overlap,
    )
    write_json(args.output, predictions)
    evidence_path = args.evidence_output or str(Path(args.output).with_suffix(".evidence.json"))
    sidecar = {
        "artifact_fingerprint": args.fingerprint,
        "data_scope": config["data_scope"],
        "device": args.device,
        "records": evidence,
    }
    if args.require_category_overlap:
        sidecar["require_category_overlap"] = True
        fallbacks = sum(1 for record in evidence if record["description_overlap_fallback"])
        sidecar["description_overlap_fallback_count"] = fallbacks
    write_json(evidence_path, sidecar)
    print(f"Wrote {len(predictions)} competition records to {args.output}")
    print(f"Wrote retrieval evidence to {evidence_path}")
    if args.require_category_overlap:
        print(f"Category-overlap description selector active; fallbacks: {fallbacks}/{len(evidence)}")


if __name__ == "__main__":
    main()
