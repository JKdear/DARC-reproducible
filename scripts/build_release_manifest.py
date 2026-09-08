#!/usr/bin/env python3
"""Build a SHA-256 inventory for every distributable release file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, npz_content_sha256, write_json

OUTPUT = "RELEASE_MANIFEST.json"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "models", "outputs"}
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


def manifest_record(path: Path) -> dict:
    """Hash .npz files by array content and every other file by exact bytes."""
    relative = path.relative_to(ROOT).as_posix()
    if path.suffix == ".npz":
        return {"path": relative, "hash_kind": "npz_content", "sha256": npz_content_sha256(path)}
    return {
        "path": relative,
        "hash_kind": "file_bytes",
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> None:
    records = [manifest_record(path) for path in distributable_files()]
    write_json(
        ROOT / OUTPUT,
        {
            "format_version": "1.1",
            "method": "DARC",
            "file_count": len(records),
            "files": records,
            "hash_policy": {
                "npz_content": "array dtype, shape, and byte digests; stable across repacking",
                "file_bytes": "exact file bytes and size",
            },
            "excluded": [
                "raw competition dataset",
                "local CLIP model directory",
                "virtual environments, caches, and .git",
                "artifacts/release_verification.json",
                OUTPUT,
            ],
        },
    )
    print(f"Wrote {len(records)} file hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
