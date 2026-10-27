"""Explicit-profile, weight-free FDG PET synthesis using fdg-pet-ct-mlx."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core.errors import InvalidInputError, MissingDependencyError, ModelExecutionError
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fdg_pet_ct_mlx._typing import FdgRuntime

MODEL = {
    "id": "fdg-pet-ct",
    "label": "Synthetic FDG PET",
    "task": "generation",
    "run": "medmlx.adapters.fdg_pet:run",
    "readiness": "medmlx.adapters.fdg_pet:readiness",
    "summary": "Weight-free PET SUV synthesis from CT, MAISI labels and explicit profiles.",
}


@dataclass(frozen=True)
class Tissue:
    suv_mean: float
    suv_std: float
    maisi_labels: tuple[int, ...]


@dataclass(frozen=True)
class Organs:
    tissues: Mapping[str, Tissue]

    def lookup_table(self, size: int) -> tuple[float, ...]:
        table = [self.tissues["unclassified_soft_tissue"].suv_mean] * size
        for tissue in self.tissues.values():
            for label in tissue.maisi_labels:
                if label < size:
                    table[label] = tissue.suv_mean
        return tuple(table)


@dataclass(frozen=True)
class Scanner:
    scatter_fraction: float
    randoms_fraction: float
    fwhm_mm: tuple[float, float, float]
    counts_per_suv: float


def _object(value: object, name: str, keys: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise InvalidInputError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if keys is not None and set(result) != keys:
        raise InvalidInputError(f"{name} must contain exactly: {', '.join(sorted(keys))}")
    return result


def _number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInputError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        domain = "positive" if positive else "nonnegative"
        raise InvalidInputError(f"{name} must be finite and {domain}")
    if maximum is not None and result > maximum:
        raise InvalidInputError(f"{name} must be at most {maximum}")
    return result


def _profiles(path: Path) -> tuple[Organs, Scanner]:
    """Parse caller-owned profiles; no tissue statistics or scanner presets are invented."""
    try:
        raw: object = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise InvalidInputError(f"Cannot read profile_path: {error}") from error
    profile = _object(raw, "profile", {"tissues", "scanner"})
    tissues: dict[str, Tissue] = {}
    assigned: set[int] = set()
    for name, raw_tissue in _object(profile["tissues"], "tissues").items():
        tissue = _object(raw_tissue, name, {"suv_mean", "suv_std", "maisi_labels"})
        raw_ids = tissue["maisi_labels"]
        if not isinstance(raw_ids, list):
            raise InvalidInputError(f"{name}.maisi_labels must be a list of integers")
        ids: list[int] = []
        for label in cast(list[object], raw_ids):
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 200:
                raise InvalidInputError("Profile MAISI label IDs must be integers in [0, 200]")
            if label in assigned:
                raise InvalidInputError(f"MAISI label {label} is assigned more than once")
            assigned.add(label)
            ids.append(label)
        tissues[name] = Tissue(
            _number(tissue["suv_mean"], f"{name}.suv_mean"),
            _number(tissue["suv_std"], f"{name}.suv_std"),
            tuple(ids),
        )
    if "unclassified_soft_tissue" not in tissues:
        raise InvalidInputError("tissues requires unclassified_soft_tissue for unlisted labels")
    scanner = _object(
        profile["scanner"],
        "scanner",
        {
            "scatter_fraction",
            "randoms_fraction",
            "fwhm_mm",
            "counts_per_suv",
        },
    )
    raw_fwhm = scanner["fwhm_mm"]
    if not isinstance(raw_fwhm, list) or len(cast(list[object], raw_fwhm)) != 3:
        raise InvalidInputError("scanner.fwhm_mm requires three axis FWHM values in mm")
    fwhm = [_number(value, "scanner.fwhm_mm") for value in cast(list[object], raw_fwhm)]
    return Organs(tissues), Scanner(
        _number(scanner["scatter_fraction"], "scatter_fraction", maximum=1),
        _number(scanner["randoms_fraction"], "randoms_fraction", maximum=1),
        (fwhm[0], fwhm[1], fwhm[2]),
        _number(scanner["counts_per_suv"], "counts_per_suv", positive=True),
    )


def run(
    *,
    image_path: str,
    labels_path: str,
    profile_path: str,
    output_dir: str,
    count_level: float = 1.0,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Synthesize PET SUV on the input CT grid with explicit organ and scanner profiles.

    No learned weights are used. The independent provider owns activity sampling
    and acquisition approximation. This adapter does not simulate projection
    physics/reconstruction, resample inputs, add lesions, or establish clinical validity.

    Args:
        image_path: Finite scalar 3-D NIfTI CT in HU, with spatial units set to mm.
        labels_path: Co-registered NIfTI MAISI labels, integer IDs in [0, 200],
            with the CT shape, affine (absolute tolerance 1e-5) and mm units.
        profile_path: JSON object with tissues and scanner. Each tissue has finite
            nonnegative suv_mean and suv_std (SUV), plus unique maisi_labels in
            [0, 200]. unclassified_soft_tissue is required for unlisted IDs.
            Scanner fields: scatter_fraction and randoms_fraction in [0, 1],
            fwhm_mm of three finite nonnegative values in array-axis order, and
            finite positive counts_per_suv. No built-in profile is assumed.
        output_dir: Write pet.nii.gz as float32 SUV, retaining the CT shape/affine.
        count_level: Finite positive multiplier on scanner counts_per_suv.
        seed: Integer in [0, 4294967294]. Activity uses seed; acquisition uses seed+1.
        overwrite: Replace pet.nii.gz only when true. Input/output aliases are refused.

    Returns:
        Absolute outputs.pet path and explicit image-space synthesis settings.
    """
    count_level = _number(count_level, "count_level", positive=True)
    if type(seed) is not int or not 0 <= seed <= 4294967294:
        raise InvalidInputError("seed must be an integer in [0, 4294967294]")
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be boolean")
    image, labels, profile = [
        Path(item).expanduser().resolve() for item in (image_path, labels_path, profile_path)
    ]
    destination = Path(output_dir).expanduser().resolve() / "pet.nii.gz"
    if destination.resolve() in {image, labels, profile}:
        raise InvalidInputError("Output must differ from all input paths")
    if destination.is_dir() or (
        (destination.exists() or destination.is_symlink()) and not overwrite
    ):
        raise InvalidInputError(f"Output exists: {destination}; pass overwrite=True to replace it")
    organs, scanner = _profiles(profile)
    if not math.isfinite(count_level * scanner.counts_per_suv):
        raise InvalidInputError("count_level * counts_per_suv must be finite")
    try:
        import nibabel as nib
    except ImportError as error:
        raise MissingDependencyError("Install medmlx's generate extra for NIfTI I/O") from error
    try:
        ct = nib.load(image)
        segmentation = nib.load(labels)
    except (OSError, ValueError) as error:
        raise InvalidInputError(f"Cannot load CT/labels NIfTI: {error}") from error
    if not isinstance(ct, nib.Nifti1Image | nib.Nifti2Image) or not isinstance(
        segmentation, nib.Nifti1Image | nib.Nifti2Image
    ):
        raise InvalidInputError("CT and labels must be NIfTI images")
    if ct.header.get_xyzt_units()[0] != "mm" or segmentation.header.get_xyzt_units()[0] != "mm":
        raise InvalidInputError("CT and labels must declare NIfTI spatial units in mm")
    if ct.affine is None or segmentation.affine is None:
        raise InvalidInputError("CT and labels require an affine")
    hu = ct.get_fdata(dtype=np.float32)
    ids = segmentation.get_fdata(dtype=np.float64)
    affine = ct.affine
    if hu.ndim != 3 or not hu.size or not np.isfinite(hu).all():
        raise InvalidInputError("CT must be a nonempty finite scalar 3-D volume in HU")
    if ids.shape != hu.shape or not np.allclose(affine, segmentation.affine, atol=1e-5, rtol=0):
        raise InvalidInputError("CT and labels must share shape and affine; no resampling occurs")
    if (
        not np.isfinite(ids).all()
        or (ids < 0).any()
        or (ids > 200).any()
        or (ids != np.floor(ids)).any()
    ):
        raise InvalidInputError("MAISI labels must be finite integer IDs in [0, 200]")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) == 0 or (spacing <= 0).any():
        raise InvalidInputError("CT affine must be finite, nonsingular and have positive spacing")
    from fdg_pet_ct_mlx.acquisition import apply_acquisition_model
    from fdg_pet_ct_mlx.activity import synthesize_activity_map
    from medmlx_core.runtime import import_mlx

    mx = cast("FdgRuntime", import_mlx())
    activity = synthesize_activity_map(mx.array(ids.astype(np.int32)), organs, seed=seed, mx=mx)
    pet = apply_acquisition_model(
        activity,
        mx.array(hu),
        scanner=scanner,
        count_level=count_level,
        spacing_mm=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        seed=seed + 1,
        mx=mx,
    )
    result: NDArray[np.float32] = np.asarray(pet, dtype=np.float32)
    if result.shape != hu.shape or not np.isfinite(result).all() or (result < 0).any():
        raise ModelExecutionError("FDG synthesis returned invalid shape or SUV values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = ct.header.copy()
    header.set_data_dtype(np.float32)
    output = type(ct)(result, affine, header)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".nii.gz",
        delete=False,
    ) as temp:
        staged = Path(temp.name)
    try:
        nib.save(output, staged)
        if overwrite:
            os.replace(staged, destination)
        else:
            os.link(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return {
        "outputs": {"pet": destination.as_posix()},
        "details": {
            "method": "image-space FDG synthesis",
            "units": "SUV",
            "seed": seed,
            "count_level": count_level,
            "shape": list(result.shape),
            "profile_path": profile.as_posix(),
            "requires_human_review": True,
        },
    }


def readiness() -> dict[str, object]:
    """Check optional packages; profiles and input contents are validated by run."""
    try:
        version("fdg-pet-ct-mlx")
        version("nibabel")
    except PackageNotFoundError as error:
        return {
            "ready": False,
            "reason": str(error),
            "setup_hint": "Install the generate extra: uv sync --locked --extra generate",
        }
    return {"ready": True, "reason": None, "setup_hint": None}
