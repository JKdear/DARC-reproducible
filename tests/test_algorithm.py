import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import evaluate, token_f1, token_jaccard
from src.features import combine_research_features, export_features
from src.runtime import (
    DARC_PARAMETERS,
    artifact_fingerprint,
    index_content_sha256,
    load_artifact,
    predict_from_embeddings,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_scope(scope):
    artifact = ROOT / f"artifacts/runtime/{scope}"
    fingerprint = (artifact / "artifact_fingerprint.txt").read_text(encoding="ascii").strip()
    return load_artifact(artifact, fingerprint)


def test_frozen_algorithm_parameters():
    assert DARC_PARAMETERS == {
        "encoder": "CLIP ViT-B/32",
        "similarity": "cosine",
        "top_k": 5,
        "category_mode": "nonnegative_similarity_weighted_relative_vote",
        "category_threshold": 0.35,
        "description_selector": "nearest_nonempty_reference",
    }


def test_prediction_uses_weighted_relative_vote_and_nearest_description():
    config = {"data_scope": "research"}
    index = {
        "embeddings": np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
        "labels": np.asarray([[1, 0], [0, 1], [0, 1]], dtype=np.uint8),
        "categories": np.asarray(["crack", "spalling"]),
        "image_ids": np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
        "reference_descriptions": np.asarray(["Nearest crack description.", "Spalling.", "Other."]),
    }
    predictions, evidence = predict_from_embeddings(
        np.asarray([[1.0, 0.0]], dtype=np.float32), ["query.jpg"], config, index
    )
    assert predictions == [
        {
            "image_id": "query.jpg",
            "damage_categories": ["crack", "spalling"],
            "description": "Nearest crack description.",
        }
    ]
    assert evidence[0]["description_source_image_id"] == "a.jpg"


def test_artifact_fingerprint_ignores_npz_container_bytes(tmp_path):
    """A repacked index with identical arrays must keep the same fingerprint."""
    source = ROOT / "artifacts/runtime/research"
    original = artifact_fingerprint(source)
    copy = tmp_path / "research"
    copy.mkdir()
    (copy / "runtime_config.json").write_bytes((source / "runtime_config.json").read_bytes())
    with np.load(source / "retrieval_index.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    np.savez(copy / "retrieval_index.npz", **arrays)

    assert (source / "retrieval_index.npz").read_bytes() != (copy / "retrieval_index.npz").read_bytes()
    assert artifact_fingerprint(copy) == original
    config, index = load_artifact(copy, original)
    assert index_content_sha256(index) == config["index_content_sha256"]
    assert index["embeddings"].dtype == np.float32


def test_artifact_fingerprint_detects_modified_arrays(tmp_path):
    source = ROOT / "artifacts/runtime/research"
    copy = tmp_path / "tampered"
    copy.mkdir()
    (copy / "runtime_config.json").write_bytes((source / "runtime_config.json").read_bytes())
    with np.load(source / "retrieval_index.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["reference_descriptions"] = arrays["reference_descriptions"].copy()
    arrays["reference_descriptions"][0] = "tampered description"
    np.savez_compressed(copy / "retrieval_index.npz", **arrays)

    assert artifact_fingerprint(copy) != artifact_fingerprint(source)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_artifact(copy, artifact_fingerprint(source))


def test_full_data_build_requires_explicit_frozen_parameter_confirmation(tmp_path):
    with pytest.raises(ValueError, match="explicit confirmation"):
        combine_research_features(
            tmp_path / "train.npz",
            tmp_path / "val.npz",
            tmp_path / "test.npz",
            tmp_path / "test_labels.npz",
            tmp_path / "provenance.json",
            tmp_path / "all_labeled.json",
            tmp_path / "output",
        )
    with pytest.raises(ValueError, match="explicit confirmation"):
        export_features(
            mode="competition",
            split_paths={"all_labeled": tmp_path / "all_labeled.json"},
            taxonomy_path=tmp_path / "taxonomy.json",
            dataset_dir=tmp_path,
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "output",
            device="cpu",
            batch_size=1,
        )


def test_both_runtime_scopes_are_integrity_checked():
    research, research_index = _load_scope("research")
    competition, competition_index = _load_scope("competition")
    assert research["sample_count"] == len(research_index["image_ids"]) == 836
    assert research["policy"]["eligible_for_original_grouped_test_evaluation"] is True
    assert competition["sample_count"] == len(competition_index["image_ids"]) == 1200
    assert competition["policy"]["eligible_for_original_grouped_test_evaluation"] is False


def test_frozen_research_metrics_match_expected_values():
    result = json.loads((ROOT / "artifacts/metrics/darc_grouped_test.json").read_text(encoding="utf-8"))
    assert result["category_metrics"]["micro_f1"] == 0.896
    assert result["category_metrics"]["primary_category_accuracy"] == pytest.approx(0.978021978021978)
    assert result["description_metrics"]["mean_token_f1"] == pytest.approx(0.6016195093949901)


def test_text_metrics_have_expected_bounds():
    assert token_f1("a crack is visible", "a crack is visible") == 1.0
    assert token_jaccard("crack concrete", "crack road") == 1 / 3
    assert token_f1("", "a crack") == 0.0


def test_independent_evaluation_requires_strict_output(tmp_path):
    truth = {
        "split": "test",
        "sample_count": 1,
        "samples": [
            {
                "image_id": "x.jpg",
                "normalized_categories": ["crack"],
                "primary_category": "crack",
                "reference_description": "A crack is visible.",
            }
        ],
    }
    predictions = [
        {
            "image_id": "x.jpg",
            "damage_categories": ["crack"],
            "description": "A crack is visible.",
            "retrieval_evidence": [],
        }
    ]
    truth_path = tmp_path / "test.json"
    prediction_path = tmp_path / "predictions.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        evaluate(truth_path, prediction_path, tmp_path / "metrics.json")
