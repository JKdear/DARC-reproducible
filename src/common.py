"""Shared deterministic I/O, hashing, and array helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_sha256(values: Iterable[str]) -> str:
    return canonical_json_sha256([str(value) for value in values])


def npz_content_sha256(path: str | Path) -> str:
    """Hash an .npz by array content so the digest survives repacking.

    Compressed archives embed metadata and depend on the local zlib build, so
    identical data yields different file bytes on different machines.
    """
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: {
                "dtype": np.asarray(payload[name]).dtype.str,
                "shape": list(np.asarray(payload[name]).shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(payload[name]).tobytes(order="C")
                ).hexdigest(),
            }
            for name in sorted(payload.files)
        }
    return canonical_json_sha256(arrays)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero-norm row")
    return array / norms


def require_lower_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text
