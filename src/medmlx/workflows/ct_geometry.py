"""Map detector RAS boxes to MedSAM2 prompts and masks back to the source grid.

NiBabel owns reorientation and resampling. Orthogonal inputs are only permuted
and flipped; oblique/sheared inputs use linear HU interpolation onto an
orthogonal RAS grid at the canonical input voxel sizes. Label restoration uses
inverse orientation or nearest neighbors. No coordinates come from the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from mlx_reason_ct.nifti import load_ct
from nibabel.affines import apply_affine, voxel_sizes
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nibabel.processing import resample_from_to, resample_to_output
from numpy.typing import NDArray

from medmlx.workflows.ct_contracts import Candidate, Detections, Measurement


@dataclass(frozen=True)
class Scan:
    """Original storage grid, physical mm grid, and the CT shared by all tools."""

    source: Path
    original: nib.Nifti1Image
    original_mm: nib.Nifti1Image
    working: nib.Nifti1Image
    working_path: Path
    resampled: bool
    assumed_mm: bool


def read_nifti(path: Path) -> nib.Nifti1Image:
    """Require a NIfTI image with an explicit affine rather than an arbitrary image type."""
    image = nib.load(path)
    if not isinstance(image, nib.Nifti1Image | nib.Nifti2Image) or image.affine is None:
        raise ValueError(f"Expected a NIfTI image with a spatial affine: {path}")
    return image


def prepare_scan(source: Path, output: Path, *, assume_mm: bool = False) -> Scan:
    """Validate HU CT geometry and save one orthogonal RAS/mm tool input.

    Unknown spatial units require the caller's explicit ``assume_mm``. The
    Reason-CT loader enforces coded, consistent qform/sform, invertibility and
    finite scalar 3D voxels. HU calibration remains an input contract.
    """
    original = read_nifti(source)
    units = original.header.get_xyzt_units()[0]
    if units == "unknown" and not assume_mm:
        raise ValueError("CT spatial units are unknown; confirm mm with assume_mm=True")
    values, affine = load_ct(source)
    if any(size < 2 for size in values.shape):
        raise ValueError("A chest CT must have at least two voxels on each spatial axis")
    original_mm = nib.Nifti1Image(values, affine)
    original_mm.header.set_xyzt_units("mm")
    canonical = nib.as_closest_canonical(original_mm)
    assert canonical.affine is not None
    diagonal = np.diag(np.diag(canonical.affine[:3, :3]))
    resampled = not np.allclose(canonical.affine[:3, :3], diagonal, atol=1e-6, rtol=0)
    working = (
        resample_to_output(canonical, voxel_sizes(canonical.affine), order=1, cval=-1024)
        if resampled
        else canonical
    )
    working.header.set_xyzt_units("mm")
    nib.save(working, output)
    # Use the serialized transform for prompts, matching the models' own loader.
    working = read_nifti(output)
    return Scan(source, original, original_mm, working, output, resampled, units == "unknown")


def ground_candidates(
    scan: Scan,
    detections_path: Path,
    *,
    threshold: float,
    limit: int,
    padding_mm: float,
) -> tuple[list[Candidate], int]:
    """Convert finite RAS centre/size boxes into native-pixel slice prompts.

    All eight corners are transformed, then clipped to voxel-center bounds.
    Boxes wholly outside the image are excluded and counted. IDs retain the
    original detection row, independent of thresholding and score sorting.
    Padding expands the propagation interval only, by physical millimetres.
    """
    document = Detections.model_validate_json(detections_path.read_text())
    if document.classes != {"0": "nodule"}:
        raise ValueError("Detector class 0 must mean nodule")
    assert scan.working.affine is not None
    inverse = np.asarray(np.linalg.inv(scan.working.affine), dtype=np.float64)
    upper = np.asarray(scan.working.shape, dtype=np.float64) - 1
    spacing = voxel_sizes(scan.working.affine)
    candidates: list[Candidate] = []
    outside = 0
    for index, row in sorted(enumerate(document.detections), key=lambda item: -item[1].score):
        if Path(row.source_image).resolve() != scan.working_path.resolve():
            raise ValueError("Detector output belongs to a different source CT")
        box = np.asarray(row.box, dtype=np.float64)
        if np.any(box[3:] <= 0):
            raise ValueError("Detector box sizes must be positive millimetres")
        if row.score < threshold:
            continue
        corners = np.asarray(
            list(product(*zip(box[:3] - box[3:] / 2, box[:3] + box[3:] / 2, strict=True)))
        )
        voxels = apply_affine(inverse, corners)
        low, high = voxels.min(axis=0), voxels.max(axis=0)
        if np.any(high < 0) or np.any(low > upper):
            outside += 1
            continue
        clipped = bool(np.any(low < 0) or np.any(high > upper))
        low, high = np.maximum(low, 0), np.minimum(high, upper)
        if np.any(high[:2] <= low[:2]):
            outside += 1
            continue
        seed = int(np.rint((low[2] + high[2]) / 2))
        start = max(0, int(np.floor(low[2] - padding_mm / spacing[2])))
        end = min(int(upper[2]), int(np.ceil(high[2] + padding_mm / spacing[2])))
        candidates.append(
            Candidate(
                candidate_id=f"candidate-{index + 1:03d}",
                detection_index=index,
                score=row.score,
                box_ras_mm=row.box,
                box_xyxy=[float(low[0]), float(low[1]), float(high[0]), float(high[1])],
                frame_idx=seed,
                slice_range=[start, end],
                clipped_to_image=clipped,
            )
        )
    return candidates[:limit], outside


def restore_mask(scan: Scan, candidate: Candidate, mask_path: Path, output: Path) -> Measurement:
    """Check a MedSAM2 binary mask and restore it to the exact source storage grid.

    Reject wrong grids, nonbinary/empty masks, propagation outside the requested
    slices, or a seed mask missing its detector prompt. Orthogonal grids use
    exact inverse permutation/flips: resampling them can lose boundary voxels
    to serialized-affine rounding, particularly after unit conversion. Oblique
    grids require nearest-neighbor restoration. Measurements use source voxels
    and the determinant of the mm affine, including anisotropy/shear.
    """
    mask = read_nifti(mask_path)
    assert mask.affine is not None and scan.working.affine is not None
    if mask.header.get_xyzt_units()[0] != "mm":
        raise ValueError("MedSAM2 mask must retain its CT input's millimetre units")
    if mask.shape != scan.working.shape or not np.allclose(
        mask.affine,
        scan.working.affine,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("MedSAM2 mask grid differs from its CT input")
    values = mask.get_fdata(dtype=np.float32)
    if not np.all(np.isin(values, [0, 1])) or not np.any(values):
        raise ValueError("MedSAM2 must return a nonempty binary mask")
    start, end = candidate.slice_range
    if np.any(values[:, :, :start]) or np.any(values[:, :, end + 1 :]):
        raise ValueError("MedSAM2 mask extends outside its requested propagation range")
    x0, y0, x1, y1 = candidate.box_xyxy
    seed = values[
        int(np.floor(x0)) : int(np.ceil(x1)) + 1,
        int(np.floor(y0)) : int(np.ceil(y1)) + 1,
        candidate.frame_idx,
    ]
    if not np.any(seed):
        raise ValueError("MedSAM2 seed mask does not overlap the detector prompt")
    assert scan.original_mm.affine is not None
    if scan.resampled:
        native = resample_from_to(mask, scan.original_mm, order=0)
        labels = np.asarray(native.get_fdata(), dtype=np.uint8)
    else:
        orientation = ornt_transform(
            io_orientation(scan.working.affine),
            io_orientation(scan.original_mm.affine),
        )
        labels = np.asarray(apply_orientation(values, orientation), dtype=np.uint8)
    indices = np.argwhere(labels != 0)
    if not len(indices):
        raise ValueError("The segmentation disappears on the source grid")
    voxel_volume = abs(float(np.linalg.det(scan.original_mm.affine[:3, :3])))
    volume = len(indices) * voxel_volume
    centroid = apply_affine(scan.original_mm.affine, indices.mean(axis=0))
    header = scan.original.header.copy()
    header.set_data_dtype(np.uint8)
    restored = type(scan.original)(labels, scan.original.affine, header)
    nib.save(restored, output)
    return Measurement(
        candidate_id=candidate.candidate_id,
        voxel_count=len(indices),
        volume_mm3=volume,
        equivalent_sphere_diameter_mm=float(np.cbrt(6 * volume / np.pi)),
        centroid_ras_mm=[float(value) for value in centroid],
        mask_path=str(output.resolve()),
    )


def display_plane[T: np.generic](values: NDArray[T], axis: int, index: int) -> NDArray[T]:
    """Use one radiological display transform for both CT and mask.

    RAS arrays display R at left in axial/coronal views, A at left in sagittal
    views, and A/S at the top respectively. Pixel aspect is handled by the viewer.
    """
    return np.take(values, index, axis=axis).T[::-1, ::-1]
