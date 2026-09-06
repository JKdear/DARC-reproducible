"""Independent grouped-test evaluation for DARC predictions."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .common import read_json, write_json
from .taxonomy import CATEGORY_DEFINITIONS, UNKNOWN_CATEGORY

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _divide(2 * precision * recall, precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_TOKEN_RE.findall((prediction or "").lower()))
    expected = Counter(_TOKEN_RE.findall((reference or "").lower()))
    if not predicted or not expected:
        return 0.0
    overlap = sum(min(predicted[token], expected[token]) for token in predicted.keys() & expected.keys())
    return _f1(_divide(overlap, sum(predicted.values())), _divide(overlap, sum(expected.values())))


def token_jaccard(prediction: str, reference: str) -> float:
    predicted = set(_TOKEN_RE.findall((prediction or "").lower()))
    expected = set(_TOKEN_RE.findall((reference or "").lower()))
    return _divide(len(predicted & expected), len(predicted | expected)) if predicted and expected else 0.0


def validate_prediction(record: dict) -> dict:
    if set(record) != {"image_id", "damage_categories", "description"}:
        raise ValueError("each prediction must contain exactly image_id, damage_categories, and description")
    image_id = record["image_id"]
    categories = record["damage_categories"]
    description = record["description"]
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("prediction image_id must be a non-empty string")
    if not isinstance(categories, list) or not categories or not all(isinstance(item, str) for item in categories):
        raise ValueError(f"prediction categories are invalid: {image_id}")
    if not isinstance(description, str):
        raise ValueError(f"prediction description is invalid: {image_id}")
    return {
        "image_id": image_id,
        "damage_categories": sorted(set(categories)),
        "description": description.strip(),
    }


def evaluate(truth_split_path: str | Path, predictions_path: str | Path, output_path: str | Path) -> dict:
    truth_payload = read_json(truth_split_path)
    if truth_payload.get("split") != "test":
        raise ValueError("formal evaluation requires the frozen test split")
    truth = truth_payload.get("samples", [])
    predictions = [validate_prediction(record) for record in read_json(predictions_path)]
    truth_by_id = {sample["image_id"]: sample for sample in truth}
    prediction_by_id = {record["image_id"]: record for record in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("predictions contain duplicate image IDs")
    if set(truth_by_id) != set(prediction_by_id):
        raise ValueError("prediction roster does not exactly match the frozen test roster")

    classes = sorted(CATEGORY_DEFINITIONS)
    stats = {category: {"tp": 0, "fp": 0, "fn": 0} for category in classes}
    exact = 0
    primary = 0
    token_scores = []
    jaccard_scores = []
    rows = []
    for image_id, sample in truth_by_id.items():
        prediction = prediction_by_id[image_id]
        expected = set(sample.get("normalized_categories", [])) - {UNKNOWN_CATEGORY}
        observed = set(prediction["damage_categories"]) - {UNKNOWN_CATEGORY}
        unknown = observed - set(classes)
        if unknown:
            raise ValueError(f"prediction contains categories outside taxonomy: {sorted(unknown)}")
        exact += expected == observed
        primary += sample.get("primary_category") in observed
        for category in classes:
            stats[category]["tp"] += category in expected and category in observed
            stats[category]["fp"] += category not in expected and category in observed
            stats[category]["fn"] += category in expected and category not in observed
        description_f1 = token_f1(prediction["description"], sample.get("reference_description", ""))
        description_jaccard = token_jaccard(prediction["description"], sample.get("reference_description", ""))
        token_scores.append(description_f1)
        jaccard_scores.append(description_jaccard)
        rows.append(
            {
                "image_id": image_id,
                "true_categories": sorted(expected),
                "predicted_categories": sorted(observed),
                "primary_category": sample.get("primary_category"),
                "category_exact_match": expected == observed,
                "description_token_f1": description_f1,
                "description_jaccard": description_jaccard,
            }
        )

    per_class = {}
    total_tp = total_fp = total_fn = 0
    for category in classes:
        values = stats[category]
        total_tp += values["tp"]
        total_fp += values["fp"]
        total_fn += values["fn"]
        precision = _divide(values["tp"], values["tp"] + values["fp"])
        recall = _divide(values["tp"], values["tp"] + values["fn"])
        per_class[category] = {
            **values,
            "true_support": values["tp"] + values["fn"],
            "predicted_support": values["tp"] + values["fp"],
            "estimable": values["tp"] + values["fn"] > 0,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    micro_precision = _divide(total_tp, total_tp + total_fp)
    micro_recall = _divide(total_tp, total_tp + total_fn)
    supported = [values["f1"] for values in per_class.values() if values["estimable"]]
    result = {
        "format_version": "1.0",
        "method": "DARC",
        "split": "test",
        "sample_count": len(truth),
        "prediction_count": len(predictions),
        "category_metrics": {
            "exact_match_accuracy": _divide(exact, len(truth)),
            "primary_category_accuracy": _divide(primary, len(truth)),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": _f1(micro_precision, micro_recall),
            "macro_f1": sum(values["f1"] for values in per_class.values()) / len(classes),
            "supported_macro_f1": sum(supported) / len(supported),
            "not_estimable_categories": [name for name, values in per_class.items() if not values["estimable"]],
            "per_class": per_class,
        },
        "description_metrics": {
            "mean_token_f1": sum(token_scores) / len(token_scores),
            "mean_jaccard": sum(jaccard_scores) / len(jaccard_scores),
        },
        "evaluated_samples": rows,
    }
    write_json(output_path, result)
    return result
