import json
from pathlib import Path

from src.common import file_sha256, sequence_sha256
from src.data import filename_group_key

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_grouped_rosters_match_public_config():
    config = json.loads((ROOT / "config/darc.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "artifacts/data/splits_grouped/split_summary.json").read_text(encoding="utf-8"))
    for name in ("train", "val", "test"):
        details = summary["splits"][name]
        assert details["sample_count"] == config["research_protocol"]["split_counts"][name]
        assert sequence_sha256(details["sample_ids"]) == config["research_protocol"]["sample_ids_sha256"][name]


def test_full_competition_roster_is_all_1200_images():
    config = json.loads((ROOT / "config/darc.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "artifacts/data/splits_competition/all_labeled.json").read_text(encoding="utf-8"))
    sample_ids = [sample["sample_id"] for sample in split["samples"]]
    image_ids = [sample["image_id"] for sample in split["samples"]]
    assert split["sample_count"] == len(sample_ids) == 1200
    assert split["eligible_for_grouped_test_evaluation"] is False
    assert sequence_sha256(sample_ids) == config["competition_protocol"]["sample_ids_sha256"]
    assert sequence_sha256(image_ids) == config["competition_protocol"]["image_ids_sha256"]


def test_sequence_grouping_matches_research_protocol():
    assert filename_group_key("crack_0012.jpg", 10) == "sequence:crack:1"
    assert filename_group_key("crack_0019.jpg", 10) == "sequence:crack:1"
    assert filename_group_key("00013.jpg", 10) == "numeric:13"


def test_imported_feature_hashes_match_allowlist():
    manifest = json.loads((ROOT / "config/server_sync_manifest.json").read_text(encoding="utf-8"))
    feature_root = ROOT / "artifacts/features/research"
    for entry in manifest["required_for_saved_research_feature_reuse"]:
        assert file_sha256(feature_root / entry["destination_name"]) == entry["sha256"]
