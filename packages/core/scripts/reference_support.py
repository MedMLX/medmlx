"""Reference-only fixture writing, with explicit upstream provenance."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, Unpack

import monai
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from fixture_cases import (  # noqa: E402
    FIXTURES,
    RECORDING_PLATFORM,
    REFERENCE,
    ArrayCase,
    HostArray,
    array_cases,
)

__all__ = ["ArrayCase", "HostArray", "array_cases", "output_directory", "save_fixture"]


def output_directory(description: str | None) -> Path:
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Reference recording requires macOS Apple Silicon")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    root = options.output_dir or FIXTURES
    if (monai.__version__, monai.__revision_id__) != (
        REFERENCE.monai_version,
        REFERENCE.monai_revision,
    ):
        raise RuntimeError("Install the pinned MONAI development dependency before recording")
    if (torch.__version__.split("+", 1)[0], torch.version.git_version) != (
        REFERENCE.torch_version,
        REFERENCE.torch_revision,
    ):
        raise RuntimeError("Install the pinned PyTorch development dependency before recording")
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    return root


class CaseRecord(TypedDict):
    id: str
    outputs: list[str]


class FixtureDetails(TypedDict, total=False):
    cases: list[CaseRecord]
    mode: str


def save_fixture(
    root: Path,
    name: str,
    seed: int,
    arrays: dict[str, HostArray],
    **details: Unpack[FixtureDetails],
) -> None:
    metadata = {
        "schema_version": 2,
        "upstream": asdict(REFERENCE),
        "platform": RECORDING_PLATFORM,
        "reference_device": "torch CPU / NumPy host",
        "numpy_version": np.__version__,
        "torch_build": torch.__version__,
        "seed": seed,
        **details,
    }
    path = root / f"{name}.npz"
    np.savez_compressed(path, allow_pickle=False, metadata=np.array(json.dumps(metadata)), **arrays)
    if path.stat().st_size >= 1_000_000:
        raise RuntimeError(f"Fixture exceeds 1 MB: {path}")
    print(f"{path}: {path.stat().st_size} bytes")
