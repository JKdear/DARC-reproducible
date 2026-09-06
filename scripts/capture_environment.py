#!/usr/bin/env python3
"""Capture the exact Python, package, accelerator, and optional model state."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_sha256, write_json
from src.features import MODEL_FILES


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "Pillow", "torch", "transformers", "pytest"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="artifacts/environment.json")
    args = parser.parse_args()
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_runtime": None,
        "nvidia_smi": None,
        "model_files": None,
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["nvidia_smi"] = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if args.model:
        model_root = Path(args.model)
        payload["model_files"] = {
            name: file_sha256(model_root / name)
            for name in MODEL_FILES
            if (model_root / name).is_file()
        }
    write_json(ROOT / args.output, payload)
    print(f"Wrote environment snapshot to {args.output}")


if __name__ == "__main__":
    main()
