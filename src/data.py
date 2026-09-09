"""Portable dataset parsing and contamination-aware grouped splitting."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .common import file_sha256, read_json, sequence_sha256, write_json
from .taxonomy import (
    UNKNOWN_CATEGORY,
    build_taxonomy_payload,
    choose_primary_category,
    extract_attributes,
    normalize_categories,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SEED = 2026
DEFAULT_WINDOW_SIZE = 10
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXPECTED_SPLIT_COUNTS = {"train": 836, "val": 182, "test": 182}
_NAME_RE = re.compile(r"^(.*?)(\d+)$")


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def load_annotations(path: str | Path) -> list[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        payload = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip().rstrip(",")
                if not text or text in {"[", "]"}:
                    continue
                try:
                    payload.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid annotation JSON on line {line_number}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("description.json must contain a JSON list")
    records = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"annotation {index} is not an object")
        normalized = {}
        for field in ("img", "prompt", "label"):
            value = record.get(field)
            if not isinstance(value, str):
                raise ValueError(f"annotation {index} has invalid {field}")
            normalized[field] = value.replace("\\", "/") if field == "img" else _normalize_text(value)
        records.append(normalized)
    return records


def _prefix(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "numeric" if stem.isdigit() else stem


def scan_images(dataset_dir: str | Path) -> list[dict]:
    root = Path(dataset_dir).resolve()
    image_dir = root / "image"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    images = []
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable image: {path}") from exc
        images.append(
            {
                "image_id": path.name,
                "image_path": f"image/{path.name}",
                "source_path": path,
                "prefix": _prefix(path.name),
                "width": width,
                "height": height,
            }
        )
    if not images:
        raise ValueError(f"no images found in {image_dir}")
    return images


def _annotation_kind(prompt: str, label: str) -> str:
    prompt_lower = prompt.lower()
    if "describe" in prompt_lower or "character" in prompt_lower or "features" in prompt_lower:
        return "description"
    if any(term in prompt_lower for term in ("determine", "judge", "does", "whether", "damage?", "damaged?")):
        return "category"
    return "description" if len(label.lower()) > 90 else "category"


def _split_annotations(annotations: list[dict]) -> tuple[dict | None, dict | None]:
    categories = []
    descriptions = []
    for annotation in annotations:
        target = descriptions if _annotation_kind(annotation["prompt"], annotation["label"]) == "description" else categories
        target.append(annotation)
    category = min(categories, key=lambda item: len(item["label"]), default=None)
    description = max(descriptions or annotations, key=lambda item: len(item["label"]), default=None)
    return category, description


def _sample_id(image_id: str) -> str:
    return hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:12]


def build_manifest(dataset_dir: str | Path) -> tuple[dict, dict[str, Path]]:
    root = Path(dataset_dir).resolve()
    annotations = load_annotations(root / "description.json")
    images = scan_images(root)
    image_by_relative = {image["image_path"]: image for image in images}
    by_stem = defaultdict(list)
    for image in images:
        by_stem[Path(image["image_path"]).stem.lower()].append(image["image_path"])

    linked = defaultdict(list)
    stem_matches = []
    missing = []
    for annotation in annotations:
        annotation_path = annotation["img"]
        if annotation_path in image_by_relative:
            linked[annotation_path].append(annotation)
            continue
        candidates = by_stem.get(Path(annotation_path).stem.lower(), [])
        if len(candidates) == 1:
            linked[candidates[0]].append(annotation)
            stem_matches.append({"annotation_path": annotation_path, "resolved_path": candidates[0]})
        else:
            missing.append(annotation_path)

    samples = []
    label_texts = []
    records_per_image = Counter()
    image_paths = {}
    for image in images:
        records = linked.get(image["image_path"], [])
        records_per_image[str(len(records))] += 1
        category_record, description_record = _split_annotations(records)
        labels = [record["label"] for record in records]
        label_texts.extend(labels)
        categories = normalize_categories(labels, prefix=image["prefix"])
        sample = {
            "sample_id": _sample_id(image["image_id"]),
            "image_id": image["image_id"],
            "image_path": image["image_path"],
            "width": image["width"],
            "height": image["height"],
            "prefix": image["prefix"],
            "category_prompt": category_record["prompt"] if category_record else "",
            "category_label_text": category_record["label"] if category_record else "",
            "description_prompt": description_record["prompt"] if description_record else "",
            "reference_description": description_record["label"] if description_record else "",
            "normalized_categories": categories,
            "primary_category": choose_primary_category(categories, prefix=image["prefix"]),
            "attributes": extract_attributes(labels),
            "annotation_count": len(records),
            "source_annotation_paths": sorted({record["img"] for record in records}),
        }
        samples.append(sample)
        image_paths[sample["sample_id"]] = image["source_path"]

    samples.sort(key=lambda sample: sample["image_id"])
    category_counts = Counter(sample["primary_category"] for sample in samples)
    manifest = {
        "format_version": "1.0",
        "method": "DARC",
        "dataset_layout": {"annotations": "description.json", "images": "image/"},
        "samples": samples,
        "stats": {
            "sample_count": len(samples),
            "annotation_count": len(annotations),
            "records_per_image": dict(sorted(records_per_image.items())),
            "missing_image_refs": sorted(set(missing)),
            "images_without_annotations": sorted(
                sample["image_id"] for sample in samples if sample["annotation_count"] == 0
            ),
            "stem_matched_image_refs": sorted(stem_matches, key=lambda item: item["annotation_path"]),
            "primary_category_counts": dict(sorted(category_counts.items())),
        },
        "taxonomy": build_taxonomy_payload(label_texts),
    }
    validate_manifest(manifest, expected_count=1200)
    return manifest, image_paths


def validate_manifest(manifest: dict, expected_count: int | None = None) -> None:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest has no samples")
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"expected {expected_count} samples, found {len(samples)}")
    image_ids = [sample.get("image_id") for sample in samples]
    sample_ids = [sample.get("sample_id") for sample in samples]
    if len(set(image_ids)) != len(samples) or len(set(sample_ids)) != len(samples):
        raise ValueError("manifest identifiers must be unique")
    for sample in samples:
        if not sample.get("reference_description") or not sample.get("normalized_categories"):
            raise ValueError(f"sample has incomplete labels: {sample.get('image_id')}")
    stats = manifest.get("stats", {})
    if stats.get("missing_image_refs") or stats.get("images_without_annotations"):
        raise ValueError("dataset contains unresolved annotation/image associations")


def filename_group_key(image_id: str, window_size: int = DEFAULT_WINDOW_SIZE) -> str:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    stem = Path(image_id).stem.lower()
    match = _NAME_RE.match(stem)
    if not match:
        return f"name:{stem}"
    prefix = match.group(1).rstrip("_- ")
    number = int(match.group(2))
    return f"sequence:{prefix}:{number // window_size}" if prefix else f"numeric:{number}"


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_groups(samples: list[dict], image_paths: dict[str, Path], window_size: int) -> list[list[int]]:
    union_find = _UnionFind(len(samples))
    sequences = defaultdict(list)
    digests = defaultdict(list)
    for index, sample in enumerate(samples):
        sequences[filename_group_key(sample["image_id"], window_size)].append(index)
        digests[file_sha256(image_paths[sample["sample_id"]])].append(index)
    for collection in (sequences, digests):
        for indices in collection.values():
            for index in indices[1:]:
                union_find.union(indices[0], index)
    groups = defaultdict(list)
    for index in range(len(samples)):
        groups[union_find.find(index)].append(index)
    return sorted(groups.values(), key=lambda group: min(samples[index]["image_id"] for index in group))


def _validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("ratios must define train, val, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must be positive and sum to one")


def assign_groups(samples: list[dict], groups: list[list[int]], seed: int) -> dict[str, list[int]]:
    ratios = dict(DEFAULT_RATIOS)
    _validate_ratios(ratios)
    names = ("train", "val", "test")
    total = len(samples)
    target_sizes = {name: total * ratios[name] for name in names}
    category_totals = Counter(sample.get("primary_category", UNKNOWN_CATEGORY) for sample in samples)
    target_categories = {
        name: {category: count * ratios[name] for category, count in category_totals.items()}
        for name in names
    }
    assignments = {name: [] for name in names}
    sizes = Counter()
    categories = {name: Counter() for name in names}
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    for group in ordered:
        group_counts = Counter(samples[index].get("primary_category", UNKNOWN_CATEGORY) for index in group)
        candidates = []
        for name in names:
            size_deficit = (target_sizes[name] - sizes[name]) / max(target_sizes[name], 1.0)
            category_deficit = sum(
                min(
                    (target_categories[name][category] - categories[name][category])
                    / max(target_categories[name][category], 1.0),
                    1.0,
                )
                * count
                for category, count in group_counts.items()
            ) / max(len(group), 1)
            overflow = max(0.0, sizes[name] + len(group) - target_sizes[name]) / max(target_sizes[name], 1.0)
            score = size_deficit + category_deficit - 4.0 * overflow
            candidates.append(((score, -sizes[name], -names.index(name)), name))
        selected = max(candidates)[1]
        assignments[selected].extend(group)
        sizes[selected] += len(group)
        categories[selected].update(group_counts)
    return assignments


def build_grouped_splits(
    manifest: dict,
    image_paths: dict[str, Path],
    seed: int = DEFAULT_SEED,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[dict, dict[str, list[dict]]]:
    samples = manifest["samples"]
    groups = build_groups(samples, image_paths, window_size)
    assignments = assign_groups(samples, groups, seed)
    splits = {
        name: sorted((samples[index] for index in indices), key=lambda sample: sample["image_id"])
        for name, indices in assignments.items()
    }
    summary = {
        "format_version": "1.0",
        "method": "DARC",
        "split_method": "group_aware_sequence_exact_duplicate",
        "seed": seed,
        "window_size": window_size,
        "near_duplicate_distance": None,
        "ratios": DEFAULT_RATIOS,
        "group_count": len(groups),
        "largest_group_size": max(map(len, groups)),
        "splits": {
            name: {
                "sample_count": len(items),
                "sample_ids": [item["sample_id"] for item in items],
                "sample_ids_sha256": sequence_sha256(item["sample_id"] for item in items),
                "image_ids_sha256": sequence_sha256(item["image_id"] for item in items),
                "primary_category_counts": dict(sorted(Counter(item["primary_category"] for item in items).items())),
            }
            for name, items in splits.items()
        },
    }
    counts = {name: len(items) for name, items in splits.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"frozen grouped split counts changed: {counts}")
    return summary, splits


def build_frozen_splits(
    manifest: dict,
    reference_split_dir: str | Path,
) -> tuple[dict, dict[str, list[dict]]]:
    """Rebuild split contents while freezing membership to image IDs.

    Annotation corrections can change category-derived grouping decisions. For a
    revised dataset whose image roster is unchanged, retaining the published
    image-ID roster keeps the formal research comparison valid.
    """
    reference_root = Path(reference_split_dir)
    samples_by_image = {sample["image_id"]: sample for sample in manifest["samples"]}
    splits: dict[str, list[dict]] = {}
    split_membership: dict[str, str] = {}

    for name in ("train", "val", "test"):
        payload = read_json(reference_root / f"{name}.json")
        reference_samples = payload.get("samples")
        if not isinstance(reference_samples, list):
            raise ValueError(f"reference split has no samples: {name}")
        image_ids = [sample.get("image_id") for sample in reference_samples]
        if any(not image_id for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
            raise ValueError(f"reference split has invalid image IDs: {name}")
        missing = sorted(set(image_ids) - set(samples_by_image))
        if missing:
            raise ValueError(f"{name} reference roster is missing from manifest: {missing[:5]}")
        duplicate_ids = set(image_ids) & set(split_membership)
        if duplicate_ids:
            raise ValueError(f"reference split rosters overlap: {sorted(duplicate_ids)[:5]}")

        items = sorted(
            (samples_by_image[image_id] for image_id in image_ids),
            key=lambda sample: sample["image_id"],
        )
        splits[name] = items
        split_membership.update({sample["image_id"]: name for sample in items})

    if set(split_membership) != set(samples_by_image):
        missing = sorted(set(samples_by_image) - set(split_membership))
        extra = sorted(set(split_membership) - set(samples_by_image))
        raise ValueError(f"frozen split roster does not cover manifest: missing={missing[:5]}, extra={extra[:5]}")

    counts = {name: len(items) for name, items in splits.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"frozen split counts changed: {counts}")

    reference_summary = read_json(reference_root / "split_summary.json")
    summary = {
        "format_version": "1.0",
        "method": "DARC",
        "split_method": reference_summary.get(
            "split_method", "group_aware_sequence_exact_duplicate"
        ),
        "seed": reference_summary.get("seed", DEFAULT_SEED),
        "window_size": reference_summary.get("window_size", DEFAULT_WINDOW_SIZE),
        "near_duplicate_distance": reference_summary.get("near_duplicate_distance"),
        "ratios": reference_summary.get("ratios", DEFAULT_RATIOS),
        "group_count": reference_summary.get("group_count"),
        "largest_group_size": reference_summary.get("largest_group_size"),
        "frozen_from": "published_image_id_roster",
        "frozen_by": "image_id",
        "splits": {
            name: {
                "sample_count": len(items),
                "sample_ids": [item["sample_id"] for item in items],
                "sample_ids_sha256": sequence_sha256(item["sample_id"] for item in items),
                "image_ids_sha256": sequence_sha256(item["image_id"] for item in items),
                "primary_category_counts": dict(
                    sorted(Counter(item["primary_category"] for item in items).items())
                ),
            }
            for name, items in splits.items()
        },
    }
    return summary, splits


def prepare_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    reference_split_dir: str | Path | None = None,
) -> dict:
    manifest, image_paths = build_manifest(dataset_dir)
    if reference_split_dir is None:
        summary, splits = build_grouped_splits(manifest, image_paths)
    else:
        summary, splits = build_frozen_splits(manifest, reference_split_dir)
    output = Path(output_dir)
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "label_taxonomy.json", manifest["taxonomy"])
    write_json(output / "splits_grouped/split_summary.json", summary)
    for name, samples in splits.items():
        write_json(output / f"splits_grouped/{name}.json", {"split": name, "sample_count": len(samples), "samples": samples})
    all_samples = list(manifest["samples"])
    write_json(
        output / "splits_competition/all_labeled.json",
        {
            "split": "all_labeled",
            "sample_count": len(all_samples),
            "use": "competition_submission_only",
            "eligible_for_grouped_test_evaluation": False,
            "samples": all_samples,
        },
    )
    return {"manifest": manifest, "summary": summary, "splits": splits}
