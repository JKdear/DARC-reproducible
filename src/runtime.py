"""Integrity-checked DARC retrieval artifact and inference core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    canonical_json_sha256,
    file_sha256,
    normalize_rows,
    read_json,
    require_lower_sha256,
    sequence_sha256,
    write_json,
)
from .features import load_features, load_split

ARTIFACT_FORMAT_VERSION = "1.0"
DARC_PARAMETERS = {
    "encoder": "CLIP ViT-B/32",
    "similarity": "cosine",
    "top_k": 5,
    "category_mode": "nonnegative_similarity_weighted_relative_vote",
    "category_threshold": 0.35,
    "description_selector": "nearest_nonempty_reference",
}
RESEARCH_PROTOCOL = {
    "split_method": "group_aware_sequence_exact_duplicate",
    "seed": 2026,
    "window_size": 10,
    "near_duplicate_distance": None,
    "split_counts": {"train": 836, "val": 182, "test": 182},
    "sample_ids_sha256": {
        "train": "4b69ef45509fc6b5ee5a2deabf40a37a53ca335f122561a5fd3c5b9ebe2fd26d",
        "val": "4c6692c701887290889c69c9b050ed8f5a5c5801ed387f59598f50eadd87ca60",
        "test": "42405870b387c6b3e7b050067b724ec457009cb8fe401b076f430ddaa1a30b46",
    },
}


def _contract_for(provenance: dict, name: str) -> dict:
    contract = provenance.get("artifacts", {}).get(name)
    if contract is None:
        contract = provenance.get("splits", {}).get(name)
    if not isinstance(contract, dict):
        raise ValueError(f"feature provenance has no {name} contract")
    return contract


def _validate_research_summary(summary: dict, samples: list[dict]) -> None:
    for key in ("split_method", "seed", "window_size", "near_duplicate_distance"):
        if summary.get(key) != RESEARCH_PROTOCOL[key]:
            raise ValueError(f"grouped research protocol mismatch: {key}")
    for name, expected_count in RESEARCH_PROTOCOL["split_counts"].items():
        details = summary.get("splits", {}).get(name, {})
        if details.get("sample_count") != expected_count:
            raise ValueError(f"grouped research count mismatch: {name}")
        sample_ids = details.get("sample_ids", [])
        if sequence_sha256(sample_ids) != RESEARCH_PROTOCOL["sample_ids_sha256"][name]:
            raise ValueError(f"grouped research sample roster mismatch: {name}")
    train_ids = [str(sample.get("sample_id") or sample["image_id"]) for sample in samples]
    if train_ids != summary["splits"]["train"]["sample_ids"]:
        raise ValueError("research index does not match grouped train roster")


def build_artifact(
    features_path: str | Path,
    split_path: str | Path,
    provenance_path: str | Path,
    output_dir: str | Path,
    data_scope: str,
    split_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a research or competition DARC retrieval artifact."""
    if data_scope not in {"research", "competition"}:
        raise ValueError("data_scope must be research or competition")
    arrays = load_features(features_path, require_labels=True)
    split_payload = read_json(split_path)
    samples = load_split(split_path)
    split_name = split_payload.get("split")
    expected_name = "train" if data_scope == "research" else "all_labeled"
    if split_name != expected_name:
        raise ValueError(f"{data_scope} artifact requires the {expected_name} split")
    expected_count = 836 if data_scope == "research" else 1200
    if len(samples) != expected_count:
        raise ValueError(f"{data_scope} artifact requires {expected_count} samples")
    if data_scope == "research":
        if split_summary_path is None:
            raise ValueError("research artifact requires --split-summary")
        _validate_research_summary(read_json(split_summary_path), samples)
    elif split_summary_path is not None:
        raise ValueError("competition artifact must not consume the research split summary")
    if data_scope == "competition":
        if split_payload.get("eligible_for_grouped_test_evaluation") is not False:
            raise ValueError("competition split must prohibit grouped-test evaluation")
        if split_payload.get("use") != "competition_submission_only":
            raise ValueError("competition split has no submission-only policy")

    image_ids = arrays["image_ids"].astype(str).tolist()
    sample_ids = arrays["sample_ids"].astype(str).tolist()
    split_image_ids = [str(sample["image_id"]) for sample in samples]
    split_sample_ids = [str(sample.get("sample_id") or sample["image_id"]) for sample in samples]
    if image_ids != split_image_ids or sample_ids != split_sample_ids:
        raise ValueError("feature order does not match artifact split")

    categories = arrays["categories"].astype(str).tolist()
    labels = np.asarray(arrays["labels"], dtype=np.uint8)
    category_index = {category: index for index, category in enumerate(categories)}
    descriptions = []
    for row, sample in enumerate(samples):
        expected = np.zeros(len(categories), dtype=np.uint8)
        for category in set(sample.get("normalized_categories", [])) - {"unknown"}:
            if category not in category_index:
                raise ValueError(f"sample category is outside artifact taxonomy: {category}")
            expected[category_index[category]] = 1
        if not np.array_equal(labels[row], expected):
            raise ValueError(f"feature labels disagree with split: {sample['image_id']}")
        description = str(sample.get("reference_description", "")).strip()
        if not description:
            raise ValueError(f"sample has no reference description: {sample['image_id']}")
        descriptions.append(description)

    provenance = read_json(provenance_path)
    if data_scope == "competition":
        policy = provenance.get("policy", {})
        if policy.get("parameters_frozen_before_full_index_build") is not True:
            raise ValueError("competition provenance does not confirm frozen parameters")
        if policy.get("eligible_for_original_grouped_test_evaluation") is not False:
            raise ValueError("competition provenance does not prohibit grouped-test evaluation")
    contract = _contract_for(provenance, expected_name)
    if contract.get("sha256") != file_sha256(features_path):
        raise ValueError("feature file does not match provenance")
    if contract.get("sample_ids_sha256") != sequence_sha256(sample_ids):
        raise ValueError("feature sample roster does not match provenance")
    if provenance.get("categories") != categories:
        raise ValueError("feature categories do not match provenance")
    model_hashes = provenance.get("local_model_file_sha256")
    if not isinstance(model_hashes, dict) or not model_hashes:
        raise ValueError("feature provenance has no pinned model files")
    for name, digest in model_hashes.items():
        require_lower_sha256(digest, f"model hash {name}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "retrieval_index.npz"
    np.savez_compressed(
        index_path,
        image_ids=np.asarray(image_ids, dtype=np.str_),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        image_sha256s=arrays["image_sha256s"].astype(np.str_),
        embeddings=normalize_rows(arrays["embeddings"]).astype(np.float32),
        labels=labels,
        categories=np.asarray(categories, dtype=np.str_),
        reference_descriptions=np.asarray(descriptions, dtype=np.str_),
    )
    config = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "method": "DARC",
        "data_scope": data_scope,
        "sample_count": expected_count,
        "embedding_dim": int(arrays["embeddings"].shape[1]),
        "categories": categories,
        "algorithm": DARC_PARAMETERS,
        "model_identifier": provenance.get("model_identifier"),
        "processor_class": provenance.get("processor_class"),
        "model_class": provenance.get("model_class"),
        "local_model_file_sha256": model_hashes,
        "index_sha256": file_sha256(index_path),
        "sample_ids_sha256": sequence_sha256(sample_ids),
        "policy": {
            "eligible_for_original_grouped_test_evaluation": data_scope == "research",
            "intended_use": (
                "reproduce_frozen_grouped_test_results"
                if data_scope == "research"
                else "predict_unseen_competition_images_only"
            ),
        },
        "sources": {
            "features_sha256": file_sha256(features_path),
            "split_sample_ids_sha256": sequence_sha256(split_sample_ids),
            "provenance_sha256": file_sha256(provenance_path),
            "split_summary_sha256": file_sha256(split_summary_path) if split_summary_path else None,
        },
    }
    unsigned = dict(config)
    config["config_payload_sha256"] = canonical_json_sha256(unsigned)
    config_path = output / "runtime_config.json"
    write_json(config_path, config)
    fingerprint = artifact_fingerprint(output)
    fingerprint_path = output / "artifact_fingerprint.txt"
    fingerprint_path.write_text(fingerprint + "\n", encoding="ascii")
    return {
        "config_path": str(config_path),
        "index_path": str(index_path),
        "fingerprint_path": str(fingerprint_path),
        "artifact_fingerprint": fingerprint,
    }


def artifact_fingerprint(artifact_dir: str | Path) -> str:
    root = Path(artifact_dir)
    return canonical_json_sha256(
        {
            "runtime_config_sha256": file_sha256(root / "runtime_config.json"),
            "retrieval_index_sha256": file_sha256(root / "retrieval_index.npz"),
        }
    )


def load_artifact(
    artifact_dir: str | Path, expected_fingerprint: str
) -> tuple[dict, dict[str, np.ndarray]]:
    require_lower_sha256(expected_fingerprint, "artifact fingerprint")
    root = Path(artifact_dir)
    if artifact_fingerprint(root) != expected_fingerprint:
        raise ValueError("runtime artifact fingerprint mismatch")
    config = read_json(root / "runtime_config.json")
    unsigned = dict(config)
    embedded_hash = unsigned.pop("config_payload_sha256", None)
    if embedded_hash != canonical_json_sha256(unsigned):
        raise ValueError("runtime config integrity check failed")
    if config.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported runtime artifact format")
    if config.get("algorithm") != DARC_PARAMETERS:
        raise ValueError("runtime artifact changes the frozen DARC algorithm")
    if config.get("data_scope") not in {"research", "competition"}:
        raise ValueError("runtime artifact has an invalid data scope")
    if file_sha256(root / "retrieval_index.npz") != config.get("index_sha256"):
        raise ValueError("retrieval index integrity check failed")
    with np.load(root / "retrieval_index.npz", allow_pickle=False) as payload:
        required = {
            "image_ids", "sample_ids", "image_sha256s", "embeddings", "labels",
            "categories", "reference_descriptions",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"retrieval index is missing arrays: {sorted(missing)}")
        index = {name: np.asarray(payload[name]) for name in required}
    count = len(index["image_ids"])
    index["embeddings"] = normalize_rows(index["embeddings"])
    index["labels"] = np.asarray(index["labels"], dtype=np.uint8)
    if count != config.get("sample_count"):
        raise ValueError("retrieval index count does not match runtime config")
    if index["embeddings"].shape != (count, config.get("embedding_dim")):
        raise ValueError("retrieval embedding shape does not match runtime config")
    if index["labels"].shape != (count, len(config.get("categories", []))):
        raise ValueError("retrieval label shape does not match runtime config")
    if index["categories"].astype(str).tolist() != config.get("categories"):
        raise ValueError("retrieval taxonomy does not match runtime config")
    return config, index


def predict_from_embeddings(
    query_embeddings: np.ndarray,
    image_ids: list[str],
    config: dict,
    index: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    query = normalize_rows(query_embeddings)
    if len(query) != len(image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("query embeddings require unique matching image IDs")
    train = index["embeddings"]
    if query.shape[1] != train.shape[1]:
        raise ValueError("query embedding dimension does not match retrieval index")
    similarities = query @ train.T
    k = min(DARC_PARAMETERS["top_k"], len(train))
    top_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
    neighbor_labels = index["labels"][top_indices]
    weights = np.clip(top_similarities, 0.0, None)
    votes = (neighbor_labels * weights[:, :, None]).sum(axis=1)
    selected = votes >= votes.max(axis=1, keepdims=True) * DARC_PARAMETERS["category_threshold"]
    empty = ~selected.any(axis=1)
    if np.any(empty):
        selected[empty, np.argmax(votes[empty], axis=1)] = True

    categories = index["categories"].astype(str).tolist()
    descriptions = index["reference_descriptions"].astype(str)
    train_image_ids = index["image_ids"].astype(str)
    predictions = []
    evidence = []
    for row, image_id in enumerate(image_ids):
        columns = np.flatnonzero(selected[row]).tolist()
        columns.sort(key=lambda column: (-float(votes[row, column]), categories[column]))
        predicted_categories = [categories[column] for column in columns]
        description_source = next(
            (neighbor for neighbor in top_indices[row] if descriptions[neighbor].strip()),
            None,
        )
        if description_source is None:
            readable = ", ".join(category.replace("_", " ") for category in predicted_categories)
            description = f"Retrieval evidence indicates visible structural damage related to {readable}."
            source_image_id = ""
        else:
            description = descriptions[description_source].strip()
            source_image_id = str(train_image_ids[description_source])
        predictions.append(
            {
                "image_id": image_id,
                "damage_categories": predicted_categories,
                "description": description,
            }
        )
        evidence.append(
            {
                "image_id": image_id,
                "data_scope": config["data_scope"],
                "description_source_image_id": source_image_id,
                "retrieval_evidence": [
                    {
                        "image_id": str(train_image_ids[neighbor]),
                        "score": float(score),
                        "damage_categories": [
                            categories[column]
                            for column in np.flatnonzero(index["labels"][neighbor])
                        ],
                    }
                    for neighbor, score in zip(top_indices[row], top_similarities[row])
                ],
            }
        )
    return predictions, evidence
