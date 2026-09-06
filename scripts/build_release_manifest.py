#!/usr/bin/env python3
"""Build a SHA-256 inventory for every distributable release file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, write_json

OUTPUT = "RELEASE_MANIFEST.json"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".venv", "models", "outputs"}
EXCLUDED_FILES = {OUTPUT, "artifacts/release_verification.json"}


def distributable_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_FILES or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if relative.startswith("data/dataset/"):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in distributable_files()
    ]
    write_json(
        ROOT / OUTPUT,
        {
            "format_version": "1.0",
            "method": "DARC",
            "file_count": len(records),
            "files": records,
            "excluded": [
                "raw competition dataset",
                "local CLIP model directory",
                "virtual environments and caches",
                "artifacts/release_verification.json",
                OUTPUT,
            ],
        },
    )
    print(f"Wrote {len(records)} file hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
