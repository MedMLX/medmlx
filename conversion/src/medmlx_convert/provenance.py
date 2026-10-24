"""Record the source checkpoint, configuration and checkout used by a recipe."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path


class SourceKind(StrEnum):
    RELEASED = "released"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class SourceRevision:
    """A checkout revision and whether local changes were present during capture."""

    repository: str
    revision: str
    dirty: bool

    @classmethod
    def capture(cls, root: Path) -> SourceRevision:
        """Record the checkout state without walking or hashing its source files."""
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                check=True, capture_output=True, text=True).stdout
        return cls(root.name, revision, bool(status))


@dataclass(frozen=True)
class BuildSource:
    """The consumed checkpoint, its release, recipe settings and installed packages."""

    kind: SourceKind
    checkpoint: str | None
    origin: str
    configuration: dict[str, str]
    recipe: str
    revision: SourceRevision
    packages: dict[str, str]

    @classmethod
    def capture(
        cls, recipe: Path, checkpoint: Path | None = None, *, origin: str = "",
        configuration: dict[str, str] | None = None, synthetic: bool = False,
    ) -> BuildSource:
        """Record source declarations and the Git checkout containing the recipe."""
        if not synthetic and (checkpoint is None or not origin):
            raise ValueError("A released build requires its checkpoint path and origin")
        # The recipe owns source identity even when the converter is installed as a wheel.
        recipe = recipe.resolve()
        checkout = subprocess.run(
            ["git", "-C", str(recipe.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        )
        root = Path(checkout.stdout.strip())
        packages = {name: version(name) for name in (
            "torch", "coreai-core", "coreai-torch", "coremltools", "numpy", "monai",
            "nnunetv2", "transformers", "keras", "h5py", "safetensors",
        )}
        kind = SourceKind.SYNTHETIC if synthetic else SourceKind.RELEASED
        return cls(kind, None if checkpoint is None else checkpoint.as_posix(), origin,
                   configuration or {}, recipe.relative_to(root).as_posix(),
                   SourceRevision.capture(root), packages)
