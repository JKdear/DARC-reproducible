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

from src.common import file_sha256, read_json


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
        source = source_root / entry["server_path"]
        if not source.is_file():
            raise SystemExit(f"missing source artifact: {source}")
        digest = file_sha256(source)
        if digest != entry["sha256"]:
            raise SystemExit(f"source SHA-256 mismatch: {source}")
        verified.append((source, destination / entry["destination_name"], digest))
    for source, target, digest in verified:
        print(f"{digest}  {source} -> {target}")
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != digest:
            raise SystemExit(f"destination SHA-256 mismatch after copy: {target}")
    action = "Verified" if args.dry_run else "Imported"
    print(f"{action} {len(verified)} allowlisted research artifacts")


if __name__ == "__main__":
    main()
