#!/usr/bin/env python3
"""Import refreshed image embeddings and regenerate labels from portable splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (
    file_sha256,
    npz_content_sha256,
    read_json,
    sequence_sha256,
    write_json,
)
from src.features import load_categories, load_split, multi_hot, validate_feature_arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--source-provenance", required=True)
    parser.add_argument("--taxonomy", default="artifacts/data/label_taxonomy.json")
    parser.add_argument("--data-artifacts", default="artifacts/data")
    parser.add_argument("--dataset", default=None, help="Optional dataset root for image digest verification.")
    parser.add_argument("--output", default="artifacts/features/research")
    parser.add_argument(
        "--changed-image-rows",
        type=int,
        default=None,
        help="Optional count of rows re-encoded during the source refresh, for provenance.",
    )
    return parser.parse_args()


def _load_source(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {"image_ids", "sample_ids", "image_sha256s", "embeddings", "categories"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return arrays


def _check_alignment(
    arrays: dict[str, np.ndarray],
    samples: list[dict],
    categories: list[str],
    source_path: Path,
    dataset_root: Path | None,
) -> None:
    image_ids = [sample["image_id"] for sample in samples]
    sample_ids = [str(sample.get("sample_id") or sample["image_id"]) for sample in samples]
    if arrays["image_ids"].astype(str).tolist() != image_ids:
        raise ValueError(f"{source_path} image order does not match its split")
    if arrays["sample_ids"].astype(str).tolist() != sample_ids:
        raise ValueError(f"{source_path} sample order does not match its split")
    if arrays["categories"].astype(str).tolist() != categories:
        raise ValueError(f"{source_path} taxonomy does not match the package taxonomy")
    validate_feature_arrays(arrays, require_labels=False)
    if dataset_root is not None:
        for sample, expected in zip(samples, arrays["image_sha256s"].astype(str)):
            image = dataset_root / sample["image_path"]
            if file_sha256(image) != expected:
                raise ValueError(f"image digest mismatch: {image}")


def _write_feature(
    path: Path,
    arrays: dict[str, np.ndarray],
    samples: list[dict],
    categories: list[str],
    include_labels: bool,
) -> dict:
    output = {
        "image_ids": np.asarray(arrays["image_ids"], dtype=np.str_),
        "sample_ids": np.asarray(arrays["sample_ids"], dtype=np.str_),
        "image_sha256s": np.asarray(arrays["image_sha256s"], dtype=np.str_),
        # Refreshed rows are already normalized; preserve their float32 bits.
        "embeddings": np.asarray(arrays["embeddings"], dtype=np.float32),
        "categories": np.asarray(categories, dtype=np.str_),
    }
    if include_labels:
        output["labels"] = multi_hot(samples, categories)
    validate_feature_arrays(output, require_labels=include_labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return {
        "path": f"artifacts/features/research/{path.name}",
        "hash_kind": "npz_content",
        "sha256": npz_content_sha256(path),
        "file_sha256": file_sha256(path),
        "sample_count": len(samples),
        "embedding_dim": int(output["embeddings"].shape[1]),
        "contains_labels": include_labels,
        "image_ids_sha256": sequence_sha256(output["image_ids"].astype(str)),
        "sample_ids_sha256": sequence_sha256(output["sample_ids"].astype(str)),
        "image_content_sha256": sequence_sha256(output["image_sha256s"].astype(str)),
    }


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_features)
    data_root = Path(args.data_artifacts)
    taxonomy_path = Path(args.taxonomy)
    dataset_root = Path(args.dataset) if args.dataset else None
    output_root = Path(args.output)
    categories = load_categories(taxonomy_path)
    source_provenance_path = Path(args.source_provenance)
    source_provenance = read_json(source_provenance_path)
    source_provenance_sha256 = file_sha256(source_provenance_path)
    model_hashes = source_provenance.get("local_model_file_sha256")
    if not isinstance(model_hashes, dict) or not model_hashes:
        raise ValueError("source provenance has no pinned model hashes")

    split_samples = {
        name: load_split(data_root / "splits_grouped" / f"{name}.json")
        for name in ("train", "val", "test")
    }
    source_arrays = {}
    source_contracts = {}
    output_contracts = {}
    for name in ("train", "val", "test"):
        source_path = source_root / f"{name}.npz"
        arrays = _load_source(source_path)
        _check_alignment(arrays, split_samples[name], categories, source_path, dataset_root)
        source_arrays[name] = arrays
        source_contracts[name] = {
            "source_label": f"refreshed_server_export/research/{name}.npz",
            "file_sha256": file_sha256(source_path),
            "content_sha256": npz_content_sha256(source_path),
        }
        output_contracts[name] = _write_feature(
            output_root / f"{name}.npz",
            arrays,
            split_samples[name],
            categories,
            include_labels=name != "test",
        )

    test_labels = {
        "image_ids": np.asarray(source_arrays["test"]["image_ids"], dtype=np.str_),
        "sample_ids": np.asarray(source_arrays["test"]["sample_ids"], dtype=np.str_),
        "labels": multi_hot(split_samples["test"], categories),
        "categories": np.asarray(categories, dtype=np.str_),
    }
    test_labels_path = output_root / "test_labels.npz"
    np.savez_compressed(test_labels_path, **test_labels)
    output_contracts["test_labels"] = {
        "path": "artifacts/features/research/test_labels.npz",
        "hash_kind": "npz_content",
        "sha256": npz_content_sha256(test_labels_path),
        "file_sha256": file_sha256(test_labels_path),
        "sample_count": len(split_samples["test"]),
        "image_ids_sha256": sequence_sha256(test_labels["image_ids"].astype(str)),
        "sample_ids_sha256": sequence_sha256(test_labels["sample_ids"].astype(str)),
    }

    if args.changed_image_rows is not None:
        if not 0 <= args.changed_image_rows <= 1200:
            raise ValueError("--changed-image-rows must be between 0 and 1200")
        refresh_scope = {
            "changed_image_rows_reencoded": args.changed_image_rows,
            "unchanged_embedding_rows_reused": 1200 - args.changed_image_rows,
        }
    else:
        refresh_scope = {}

    provenance = {
        "format_version": "1.0",
        "method": "DARC",
        "construction": "refreshed_image_embeddings_with_labels_regenerated_from_updated_manifest",
        "model_identifier": "openai/clip-vit-base-patch32",
        "processor_class": source_provenance.get("processor_class", "CLIPProcessor"),
        "model_class": source_provenance.get("model_class", "CLIPModel"),
        "local_model_file_sha256": model_hashes,
        "model_input": "image pixels only; filenames and annotations are not passed to CLIP",
        "categories": categories,
        "taxonomy_path": "artifacts/data/label_taxonomy.json",
        "taxonomy_sha256": file_sha256(taxonomy_path),
        "runtime": source_provenance.get("runtime", {}),
        "artifacts": output_contracts,
        "source_refresh": {
            "source_provenance_sha256": source_provenance_sha256,
            "source_files": source_contracts,
            **refresh_scope,
        },
        "policy": {
            "parameters_frozen_before_full_index_build": False,
            "eligible_for_original_grouped_test_evaluation": True,
            "intended_use": "frozen_grouped_research_evaluation",
        },
    }
    write_json(output_root / "provenance.json", provenance)
    print(f"Imported refreshed embeddings and regenerated labels for {sum(len(items) for items in split_samples.values())} images")
    print(f"Wrote research features to {output_root}")


if __name__ == "__main__":
    main()
