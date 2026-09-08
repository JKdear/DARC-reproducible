#!/usr/bin/env python3
"""Verify DARC code, frozen results, artifacts, and optional CLIP files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, npz_content_sha256, read_json, sequence_sha256, write_json
from src.runtime import DARC_PARAMETERS, load_artifact


def _record(checks: dict, name: str, function) -> None:
    try:
        detail = function()
        checks[name] = {"passed": True, "detail": detail}
    except Exception as exc:  # The report must retain all independent failures.
        checks[name] = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Optional local CLIP directory to verify")
    parser.add_argument("--output", default="artifacts/release_verification.json")
    args = parser.parse_args()
    configuration = read_json(ROOT / "config/darc.json")
    checks = {}

    def verify_configuration():
        if configuration["algorithm"] != DARC_PARAMETERS:
            raise ValueError("config/darc.json differs from the implementation")
        return "frozen DARC algorithm matches implementation"

    def verify_data():
        summary = read_json(ROOT / "artifacts/data/splits_grouped/split_summary.json")
        expected = configuration["research_protocol"]
        counts = {name: summary["splits"][name]["sample_count"] for name in ("train", "val", "test")}
        if counts != expected["split_counts"]:
            raise ValueError(f"split counts differ: {counts}")
        for name, digest in expected["sample_ids_sha256"].items():
            if sequence_sha256(summary["splits"][name]["sample_ids"]) != digest:
                raise ValueError(f"{name} sample roster differs")
        full = read_json(ROOT / "artifacts/data/splits_competition/all_labeled.json")
        full_ids = [sample["sample_id"] for sample in full["samples"]]
        if len(full_ids) != configuration["competition_protocol"]["labeled_index_count"]:
            raise ValueError("competition roster does not contain 1200 samples")
        if sequence_sha256(full_ids) != configuration["competition_protocol"]["sample_ids_sha256"]:
            raise ValueError("competition sample roster differs")
        return {"grouped_counts": counts, "competition_count": len(full_ids)}

    def verify_imported_features():
        manifest = read_json(ROOT / "config/server_sync_manifest.json")
        source = ROOT / "artifacts/features/research"
        verified = 0
        for entry in manifest["required_for_saved_research_feature_reuse"]:
            path = source / entry["destination_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"feature hash mismatch: {path.name}")
            verified += 1
        return f"{verified} frozen research feature files match server hashes"

    def verify_runtime(scope: str):
        root = ROOT / f"artifacts/runtime/{scope}"
        fingerprint = (root / "artifact_fingerprint.txt").read_text(encoding="ascii").strip()
        config, _ = load_artifact(root, fingerprint)
        if config["data_scope"] != scope:
            raise ValueError("runtime scope mismatch")
        expected_count = 836 if scope == "research" else 1200
        if config["sample_count"] != expected_count:
            raise ValueError("runtime sample count mismatch")
        if scope == "competition" and config["policy"]["eligible_for_original_grouped_test_evaluation"]:
            raise ValueError("competition runtime incorrectly permits grouped-test evaluation")
        return {"fingerprint": fingerprint, "sample_count": expected_count}

    def verify_metrics():
        result = read_json(ROOT / "artifacts/metrics/darc_grouped_test.json")
        expected = configuration["research_protocol"]["expected_metrics"]
        observed = {
            "exact_match_accuracy": result["category_metrics"]["exact_match_accuracy"],
            "primary_category_accuracy": result["category_metrics"]["primary_category_accuracy"],
            "micro_f1": result["category_metrics"]["micro_f1"],
            "macro_f1_fixed_11_classes": result["category_metrics"]["macro_f1"],
            "mean_description_token_f1": result["description_metrics"]["mean_token_f1"],
            "mean_description_jaccard": result["description_metrics"]["mean_jaccard"],
        }
        for name, expected_value in expected.items():
            if not math.isclose(observed[name], expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"metric differs: {name}")
        return observed

    def verify_predictions():
        predictions = read_json(ROOT / "artifacts/predictions/darc_grouped_test.json")
        if len(predictions) != 182:
            raise ValueError("research prediction count is not 182")
        required = {"image_id", "damage_categories", "description"}
        if any(set(record) != required for record in predictions):
            raise ValueError("prediction JSON violates the strict competition schema")
        return "182 strict prediction records"

    def verify_release_manifest():
        inventory = read_json(ROOT / "RELEASE_MANIFEST.json")
        records = inventory.get("files", [])
        if inventory.get("file_count") != len(records) or not records:
            raise ValueError("release manifest has an invalid file count")
        for record in records:
            path = ROOT / record["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if record.get("hash_kind") == "npz_content":
                observed = npz_content_sha256(path)
            else:
                if path.stat().st_size != record["size_bytes"]:
                    raise ValueError(f"release file size differs: {record['path']}")
                observed = file_sha256(path)
            if observed != record["sha256"]:
                raise ValueError(f"release file differs: {record['path']}")
        kinds = {record.get("hash_kind", "file_bytes") for record in records}
        return f"{len(records)} distributable files match ({', '.join(sorted(kinds))})"

    def verify_model():
        if not args.model:
            return "not requested"
        model_root = Path(args.model)
        expected = configuration["clip_model_files"]
        for name, digest in expected.items():
            path = model_root / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"model file missing or mismatched: {name}")
        return f"{len(expected)} pinned CLIP files verified"

    _record(checks, "configuration", verify_configuration)
    _record(checks, "data_protocol", verify_data)
    _record(checks, "frozen_research_features", verify_imported_features)
    _record(checks, "research_runtime", lambda: verify_runtime("research"))
    _record(checks, "competition_runtime", lambda: verify_runtime("competition"))
    _record(checks, "research_predictions", verify_predictions)
    _record(checks, "research_metrics", verify_metrics)
    _record(checks, "release_manifest", verify_release_manifest)
    _record(checks, "local_clip_model", verify_model)
    passed = all(item["passed"] for item in checks.values())
    report = {"format_version": "1.0", "method": "DARC", "passed": passed, "checks": checks}
    write_json(ROOT / args.output, report)
    for name, result in checks.items():
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}: {result['detail']}")
    print(f"Wrote verification report to {args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
