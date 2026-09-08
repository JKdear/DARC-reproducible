"""Frozen CLIP feature export and research-feature consolidation."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    canonical_json_sha256,
    file_sha256,
    normalize_rows,
    npz_content_sha256,
    read_json,
    sequence_sha256,
    write_json,
)

FEATURE_FORMAT_VERSION = "1.0"
MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "pytorch_model.bin",
)


def load_categories(taxonomy_path: str | Path) -> list[str]:
    payload = read_json(taxonomy_path)
    categories = [item.get("name") for item in payload.get("categories", []) if item.get("name")]
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("taxonomy must contain unique categories")
    return categories


def load_split(path: str | Path) -> list[dict]:
    payload = read_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"split has no samples: {path}")
    if payload.get("sample_count") != len(samples):
        raise ValueError(f"split sample_count mismatch: {path}")
    return samples


def image_path(sample: dict, dataset_dir: str | Path) -> Path:
    relative = sample.get("image_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"sample has no image_path: {sample.get('image_id')}")
    path = Path(dataset_dir) / relative
    if not path.is_file():
        raise FileNotFoundError(f"image is missing: {path}")
    return path


def multi_hot(samples: list[dict], categories: list[str]) -> np.ndarray:
    category_index = {category: index for index, category in enumerate(categories)}
    labels = np.zeros((len(samples), len(categories)), dtype=np.uint8)
    for row, sample in enumerate(samples):
        observed = set(sample.get("normalized_categories", [])) - {"unknown"}
        unknown = observed - set(category_index)
        if unknown:
            raise ValueError(f"sample has categories outside taxonomy: {sorted(unknown)}")
        if not observed:
            raise ValueError(f"sample has no supported labels: {sample.get('image_id')}")
        for category in observed:
            labels[row, category_index[category]] = 1
    return labels


def model_file_hashes(model_dir: str | Path) -> dict[str, str]:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError("DARC requires a local CLIP model directory")
    hashes = {name: file_sha256(root / name) for name in MODEL_FILES if (root / name).is_file()}
    if "config.json" not in hashes or not ({"model.safetensors", "pytorch_model.bin"} & set(hashes)):
        raise ValueError("local CLIP directory is missing config or model weights")
    return hashes


def _identity_arrays(samples: list[dict], dataset_dir: str | Path) -> dict[str, np.ndarray]:
    return {
        "image_ids": np.asarray([sample["image_id"] for sample in samples], dtype=np.str_),
        "sample_ids": np.asarray([sample.get("sample_id") or sample["image_id"] for sample in samples], dtype=np.str_),
        "image_sha256s": np.asarray([file_sha256(image_path(sample, dataset_dir)) for sample in samples], dtype=np.str_),
    }


def validate_feature_arrays(arrays: dict[str, np.ndarray], require_labels: bool) -> None:
    required = {"image_ids", "sample_ids", "image_sha256s", "embeddings", "categories"}
    if require_labels:
        required.add("labels")
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"feature file is missing arrays: {sorted(missing)}")
    count = len(arrays["image_ids"])
    if count == 0 or len(arrays["sample_ids"]) != count or len(arrays["image_sha256s"]) != count:
        raise ValueError("feature identity arrays have inconsistent lengths")
    if len(set(arrays["image_ids"].astype(str))) != count or len(set(arrays["sample_ids"].astype(str))) != count:
        raise ValueError("feature identifiers must be unique")
    if any(len(value) != 64 for value in arrays["image_sha256s"].astype(str)):
        raise ValueError("image_sha256s contains an invalid digest")
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != count or not np.isfinite(embeddings).all():
        raise ValueError("embeddings have an invalid shape or value")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("embeddings are not L2-normalized")
    categories = arrays["categories"].astype(str)
    if len(categories) == 0 or len(set(categories)) != len(categories):
        raise ValueError("feature categories must be non-empty and unique")
    if require_labels:
        labels = np.asarray(arrays["labels"])
        if labels.shape != (count, len(categories)) or not np.isin(labels, (0, 1)).all():
            raise ValueError("labels have an invalid shape or value")
        if np.any(labels.sum(axis=1) == 0):
            raise ValueError("every labeled feature row needs a positive class")


def load_features(path: str | Path, require_labels: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    validate_feature_arrays(arrays, require_labels=require_labels)
    if not require_labels and "labels" in arrays:
        raise ValueError("unlabeled query features must not contain labels")
    return arrays


def save_features(
    path: str | Path,
    samples: list[dict],
    identities: dict[str, np.ndarray],
    embeddings: np.ndarray,
    categories: list[str],
    include_labels: bool,
) -> dict[str, Any]:
    arrays = {
        **identities,
        "embeddings": normalize_rows(embeddings).astype(np.float32),
        "categories": np.asarray(categories, dtype=np.str_),
    }
    if include_labels:
        arrays["labels"] = multi_hot(samples, categories)
    validate_feature_arrays(arrays, require_labels=include_labels)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {
        "path": destination.as_posix(),
        "hash_kind": "npz_content",
        "sha256": npz_content_sha256(destination),
        "sample_count": len(samples),
        "embedding_dim": int(arrays["embeddings"].shape[1]),
        "contains_labels": include_labels,
        "image_ids_sha256": sequence_sha256(arrays["image_ids"].astype(str)),
        "sample_ids_sha256": sequence_sha256(arrays["sample_ids"].astype(str)),
        "image_content_sha256": sequence_sha256(arrays["image_sha256s"].astype(str)),
    }


def _load_clip(model_dir: str | Path, device: str):
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("feature export requires torch, transformers, and Pillow") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    model.eval()
    return torch, transformers, Image, processor, model


def _encode(
    samples: list[dict],
    dataset_dir: str | Path,
    processor,
    model,
    torch,
    image_class,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = []
    for start in range(0, len(samples), batch_size):
        paths = [image_path(sample, dataset_dir) for sample in samples[start : start + batch_size]]
        images = []
        try:
            for path in paths:
                with image_class.open(path) as image:
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
        print(f"Encoded {min(start + batch_size, len(samples))}/{len(samples)} images")
    return np.concatenate(batches, axis=0)


def export_features(
    mode: str,
    split_paths: dict[str, str | Path],
    taxonomy_path: str | Path,
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int,
    parameters_frozen: bool = False,
) -> dict:
    if mode not in {"research", "competition"}:
        raise ValueError("mode must be research or competition")
    if mode == "competition" and not parameters_frozen:
        raise ValueError("competition export requires explicit confirmation that parameters are frozen")
    required_names = {"train", "val", "test"} if mode == "research" else {"all_labeled"}
    if set(split_paths) != required_names:
        raise ValueError(f"{mode} mode requires split names {sorted(required_names)}")
    categories = load_categories(taxonomy_path)
    hashes = model_file_hashes(model_dir)
    torch, transformers, image_class, processor, model = _load_clip(model_dir, device)
    output = Path(output_dir)
    samples_by_name = {name: load_split(path) for name, path in split_paths.items()}
    identities = {
        name: _identity_arrays(samples, dataset_dir) for name, samples in samples_by_name.items()
    }
    if mode == "research":
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            for field in ("image_ids", "sample_ids", "image_sha256s"):
                overlap = set(identities[left][field].astype(str)) & set(identities[right][field].astype(str))
                if overlap:
                    raise ValueError(f"{left}/{right} {field} overlap detected")

    provenance = {
        "format_version": FEATURE_FORMAT_VERSION,
        "method": "DARC",
        "mode": mode,
        "model_identifier": "openai/clip-vit-base-patch32",
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "local_model_file_sha256": hashes,
        "categories": categories,
        "taxonomy_sha256": file_sha256(taxonomy_path),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": device,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "artifacts": {},
        "policy": {
            "parameters_frozen_before_full_index_build": mode == "competition",
            "eligible_for_original_grouped_test_evaluation": mode == "research",
            "intended_use": (
                "frozen_grouped_research_evaluation"
                if mode == "research"
                else "prediction_on_unseen_competition_images_only"
            ),
        },
    }
    for name in sorted(samples_by_name):
        samples = samples_by_name[name]
        embeddings = _encode(samples, dataset_dir, processor, model, torch, image_class, device, batch_size)
        include_labels = name != "test"
        details = save_features(
            output / f"{name}.npz",
            samples,
            identities[name],
            embeddings,
            categories,
            include_labels=include_labels,
        )
        details["split_sha256"] = file_sha256(split_paths[name])
        provenance["artifacts"][name] = details
        if name == "test":
            labels_path = output / "test_labels.npz"
            np.savez_compressed(
                labels_path,
                image_ids=identities[name]["image_ids"],
                sample_ids=identities[name]["sample_ids"],
                labels=multi_hot(samples, categories),
                categories=np.asarray(categories, dtype=np.str_),
            )
            provenance["artifacts"]["test_labels"] = {
                "path": labels_path.as_posix(),
                "hash_kind": "npz_content",
                "sha256": npz_content_sha256(labels_path),
                "sample_count": len(samples),
                "image_ids_sha256": sequence_sha256(identities[name]["image_ids"].astype(str)),
                "sample_ids_sha256": sequence_sha256(identities[name]["sample_ids"].astype(str)),
            }
    write_json(output / "provenance.json", provenance)
    return provenance


def combine_research_features(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    test_labels_path: str | Path,
    research_provenance_path: str | Path,
    all_labeled_split_path: str | Path,
    output_dir: str | Path,
    parameters_frozen: bool = False,
) -> dict:
    """Combine frozen disjoint features after model selection for final submission."""
    if not parameters_frozen:
        raise ValueError("full-data combination requires explicit confirmation that parameters are frozen")
    sources = {
        "train": load_features(train_path, require_labels=True),
        "val": load_features(val_path, require_labels=True),
        "test": load_features(test_path, require_labels=False),
    }
    with np.load(test_labels_path, allow_pickle=False) as payload:
        required = {"image_ids", "sample_ids", "labels", "categories"}
        if required - set(payload.files):
            raise ValueError("test label file is incomplete")
        test_labels = {name: np.asarray(payload[name]) for name in required}
    if not np.array_equal(test_labels["image_ids"].astype(str), sources["test"]["image_ids"].astype(str)):
        raise ValueError("test label image IDs do not match test features")
    if not np.array_equal(test_labels["sample_ids"].astype(str), sources["test"]["sample_ids"].astype(str)):
        raise ValueError("test label sample IDs do not match test features")
    sources["test"]["labels"] = np.asarray(test_labels["labels"], dtype=np.uint8)
    categories = sources["train"]["categories"].astype(str)
    for name, arrays in sources.items():
        if not np.array_equal(arrays["categories"].astype(str), categories):
            raise ValueError(f"{name} categories differ from train categories")
        validate_feature_arrays(arrays, require_labels=True)

    rows = {}
    for split_name, arrays in sources.items():
        for index, sample_id in enumerate(arrays["sample_ids"].astype(str)):
            if sample_id in rows:
                raise ValueError(f"duplicate sample across research splits: {sample_id}")
            rows[sample_id] = {name: arrays[name][index] for name in ("image_ids", "sample_ids", "image_sha256s", "embeddings", "labels")}
    all_samples = load_split(all_labeled_split_path)
    expected_ids = [str(sample.get("sample_id") or sample["image_id"]) for sample in all_samples]
    if set(rows) != set(expected_ids) or len(expected_ids) != 1200:
        raise ValueError("research features do not cover the 1200-sample competition split")
    combined = {
        name: np.asarray([rows[sample_id][name] for sample_id in expected_ids])
        for name in ("image_ids", "sample_ids", "image_sha256s", "embeddings", "labels")
    }
    # Rows are copied verbatim from already-normalized research features; a second
    # normalization would perturb float bits and break cross-machine identity.
    combined["embeddings"] = np.asarray(combined["embeddings"], dtype=np.float32)
    combined["labels"] = combined["labels"].astype(np.uint8)
    combined["categories"] = categories.astype(np.str_)
    validate_feature_arrays(combined, require_labels=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "all_labeled.npz"
    np.savez_compressed(destination, **combined)
    research_provenance = read_json(research_provenance_path)
    expected_model_hashes = research_provenance.get("local_model_file_sha256")
    if not isinstance(expected_model_hashes, dict) or not expected_model_hashes:
        raise ValueError("research provenance has no model file hashes")
    provenance = {
        "format_version": FEATURE_FORMAT_VERSION,
        "method": "DARC",
        "mode": "competition",
        "construction": "combined_from_frozen_research_features_after_model_selection",
        "model_identifier": research_provenance.get("model_identifier"),
        "processor_class": research_provenance.get("processor_class"),
        "model_class": research_provenance.get("model_class"),
        "local_model_file_sha256": expected_model_hashes,
        "categories": categories.tolist(),
        "artifacts": {
            "all_labeled": {
                "path": destination.as_posix(),
                "hash_kind": "npz_content",
                "sha256": npz_content_sha256(destination),
                "sample_count": len(expected_ids),
                "embedding_dim": int(combined["embeddings"].shape[1]),
                "contains_labels": True,
                "image_ids_sha256": sequence_sha256(combined["image_ids"].astype(str)),
                "sample_ids_sha256": sequence_sha256(combined["sample_ids"].astype(str)),
                "image_content_sha256": sequence_sha256(combined["image_sha256s"].astype(str)),
                "split_sha256": file_sha256(all_labeled_split_path),
            }
        },
        # Source .npz files are hashed by array content so this provenance file is
        # byte-identical on every machine that combines the same frozen inputs.
        "sources": {
            "train_content_sha256": npz_content_sha256(train_path),
            "val_content_sha256": npz_content_sha256(val_path),
            "test_content_sha256": npz_content_sha256(test_path),
            "test_labels_content_sha256": npz_content_sha256(test_labels_path),
            "research_provenance_sha256": file_sha256(research_provenance_path),
        },
        "policy": {
            "parameters_frozen_before_full_index_build": True,
            "eligible_for_original_grouped_test_evaluation": False,
            "intended_use": "prediction_on_unseen_competition_images_only",
        },
    }
    write_json(output / "provenance.json", provenance)
    return provenance
