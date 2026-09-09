#!/usr/bin/env python3
"""Import the allowlisted server artifacts only after SHA-256 verification."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, npz_content_sha256, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Server checkout or verified staging root")
    parser.add_argument("--manifest", default="config/server_sync_manifest.json")
    parser.add_argument("--destination", default="artifacts/features/research")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    source_root = Path(args.source_root)
    destination = Path(args.destination)
    entries = manifest["required_for_saved_research_feature_reuse"]
    verified = []
    for entry in entries:
        source_name = entry.get("source_path", entry.get("server_path"))
        if not source_name:
            raise SystemExit("manifest entry has no source_path")
        source = source_root / source_name
        if not source.is_file():
            raise SystemExit(f"missing source artifact: {source}")
        hash_kind = entry.get("hash_kind", "file_bytes")
        digest = npz_content_sha256(source) if hash_kind == "npz_content" else file_sha256(source)
        if digest != entry["sha256"]:
            raise SystemExit(f"source SHA-256 mismatch: {source}")
        verified.append((source, destination / entry["destination_name"], digest, hash_kind))
    for source, target, digest, hash_kind in verified:
        print(f"{digest}  {source} -> {target}")
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        observed = npz_content_sha256(target) if hash_kind == "npz_content" else file_sha256(target)
        if observed != digest:
            raise SystemExit(f"destination SHA-256 mismatch after copy: {target}")
    action = "Verified" if args.dry_run else "Imported"
    print(f"{action} {len(verified)} allowlisted research artifacts")


if __name__ == "__main__":
    main()
