"""Read model inputs and outputs for clients: DICOM conversion, NIfTI summaries, previews.

These helpers let an agent go from a DICOM folder to a model input, and from a
model's output mask to label volumes and a picture. Summaries and previews need
nibabel and Pillow, and DICOM conversion needs SimpleITK; all three come with
``medmlx[all]``, and SimpleITK alone with ``medmlx[dicom]``. They are imported
only when a helper is called.

Nothing here interprets findings. Volumes are voxel counts times the affine's
voxel volume, and previews are display renderings of one slice.
"""

from __future__ import annotations

import io
import json
import os
import textwrap
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from medmlx.payloads import (
    ConvertedSeries,
    DicomSeries,
    LabelSummary,
    OutputNamesFrom,
    OutputSummary,
)

if TYPE_CHECKING:
    import nibabel
    import numpy as np
    from numpy.typing import NDArray

# Integer masks with more distinct values than this are treated as images.
_MAX_LABELS = 512
_PREVIEW_SIZE = 512
_CT_WINDOW = (-160.0, 240.0)
# Distinct, muted colors that read on a gray scan; label n uses entry n % len.
_PALETTE = (
    (230, 97, 92), (93, 165, 218), (250, 164, 58), (96, 189, 104), (241, 124, 176),
    (178, 145, 47), (178, 118, 178), (222, 207, 63), (77, 201, 201), (241, 88, 84),
    (140, 140, 220), (160, 200, 120),
)
_MISSING = "{name} is required; install medmlx[{extra}]"
# DICOM objects that are not images: segmentations, structure sets, reports, plans.
_NOT_IMAGES = frozenset(
    {"SEG", "RTSTRUCT", "RTPLAN", "RTDOSE", "RTRECORD", "SR", "PR", "KO", "REG"},
)


def _resolve(path: str | Path, root: str | Path | None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and root is not None:
        candidate = Path(root) / candidate
    return candidate.resolve()


def _nifti(path: Path) -> nibabel.Nifti1Image:
    """Load a NIfTI file reoriented to RAS, or raise a client-readable error."""
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError(_MISSING.format(name="nibabel", extra="all")) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(path)
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError(f"{path} is not a NIfTI image")
    return nib.as_closest_canonical(image)


def _volume(image: nibabel.Nifti1Image) -> NDArray[np.generic]:
    """Return the image's array as at most three dimensions, keeping the first frame."""
    import numpy as np

    data = np.asanyarray(image.dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    return data[:, :, None] if data.ndim == 2 else data


def _millimetres_per_unit(image: nibabel.Nifti1Image) -> float:
    """Convert declared spatial units; retain the existing mm assumption when unspecified."""
    return {"meter": 1000.0, "mm": 1.0, "micron": 0.001, "unknown": 1.0}[
        image.header.get_xyzt_units()[0]
    ]


def summarize_output(
    path: str | Path, *, labels: Mapping[int, str] | None = None, root: str | Path | None = None,
) -> OutputSummary:
    """Summarize a NIfTI output: label voxel counts and volumes, or an image's value range.

    Integer-valued volumes with at most 512 distinct values are treated as label
    masks; every nonzero label is reported with its voxel count and volume in
    millilitres from the affine's voxel volume. ``labels`` supplies names, such as
    a model's declared label map. Other volumes report their value range.
    Declared spatial units are converted to millimetres; unspecified units are
    treated as millimetres.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is not a NIfTI image.
        ImportError: nibabel is not installed.
    """
    import numpy as np

    file = _resolve(path, root)
    image = _nifti(file)
    data = _volume(image)
    scale = _millimetres_per_unit(image)
    zooms = [float(size) * scale for size in image.header.get_zooms()[:3]]
    affine = image.affine
    if affine is None:
        raise ValueError(f"{file} has no spatial affine")
    # Multiplying axis lengths overstates voxel volume when the affine has shear.
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) * scale**3 / 1000.0
    summary: OutputSummary = {
        "path": file.as_posix(), "shape": [int(size) for size in data.shape],
        "spacing_mm": zooms, "dtype": str(data.dtype), "kind": "image",
        "labels": [], "value_range": None, "names_from": None,
    }
    integral = bool(np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool_))
    if not integral and np.issubdtype(data.dtype, np.floating) and data.size:
        values = data.astype(np.float64)
        integral = bool(np.all(np.isfinite(values)) and np.array_equal(values, np.round(values)))
    if integral:
        if np.issubdtype(data.dtype, np.unsignedinteger) and data.dtype.itemsize <= 2:
            # At most 65536 bins: avoid sorting a widened copy of a small-label mask.
            histogram = np.bincount(data.reshape(-1))
            values = np.flatnonzero(histogram)
            counts = histogram[values]
        else:
            # Keep wide unsigned and floating labels intact rather than narrowing to int64.
            count_data = (
                data.astype(np.int64, copy=False) if np.can_cast(data.dtype, np.int64) else data
            )
            values, counts = np.unique(count_data, return_counts=True)
        if len(values) <= _MAX_LABELS:
            names = dict(labels or {})
            rows: list[LabelSummary] = [
                {"value": int(value), "name": names.get(int(value)), "voxels": int(count),
                 "volume_ml": round(int(count) * voxel_ml, 3)}
                for value, count in zip(values.tolist(), counts.tolist(), strict=True)
                if value != 0
            ]
            summary["kind"] = "labels"
            summary["labels"] = rows
            return summary
    if data.size:
        values = data.astype(np.float64)
        summary["value_range"] = [float(np.nanmin(values)), float(np.nanmax(values))]
    return summary


def _display(plane: NDArray[np.generic]) -> NDArray[np.generic]:
    """Show an RAS axial plane radiologically: anterior up, patient right on the left."""
    return plane[::-1, ::-1].T


def preview_output(
    path: str | Path, *, image_path: str | Path | None = None, root: str | Path | None = None,
    labels: Mapping[int, str] | None = None, names_from: OutputNamesFrom = None,
) -> bytes:
    """Render one axial slice of a NIfTI output as a PNG, over its scan when given.

    Volumes are reoriented to RAS and shown radiologically: patient right on the
    image left, anterior at the top. For a label mask the slice with the most
    labeled voxels is shown, colored by label; otherwise the middle slice. A
    scan whose values reach below -500 is shown in a soft-tissue CT window
    (-160 to 240 HU); other scans use their 1st to 99th percentiles.
    Overlays require matching shapes and canonical affines within 1e-5 mm; images
    on different physical grids must be registered or resampled first.
    Named labels appear in a legend. PNG text fields contain ``labels`` as JSON
    and ``names_from`` as ``outputs``, ``declaration`` or ``null``.

    Raises:
        FileNotFoundError: A path does not exist.
        ValueError: A file is not NIfTI, or the output and scan grids differ.
        ImportError: nibabel or Pillow is not installed.
    """
    import numpy as np

    try:
        from PIL import Image, ImageDraw, PngImagePlugin
    except ImportError as error:
        raise ImportError(_MISSING.format(name="Pillow", extra="all")) from error
    output = _resolve(path, root)
    volume = _nifti(output)
    data = _volume(volume)
    summary = summarize_output(output, labels=labels)
    mask = summary["kind"] == "labels"
    index = int(np.argmax((data != 0).sum(axis=(0, 1)))) if mask else data.shape[2] // 2
    label_plane = _display(data[:, :, index]).astype(np.int64) if mask else None
    if image_path is not None:
        scan_image = _nifti(_resolve(image_path, root))
        scan = _volume(scan_image)
        if scan.shape != data.shape:
            raise ValueError(
                f"Output grid {list(data.shape)} differs from scan grid {list(scan.shape)}")
        output_affine, scan_affine = volume.affine, scan_image.affine
        if output_affine is None or scan_affine is None or not np.allclose(
            output_affine[:3] * _millimetres_per_unit(volume),
            scan_affine[:3] * _millimetres_per_unit(scan_image), rtol=0, atol=1e-5,
        ):
            raise ValueError("Output affine differs from scan affine; overlay grids must match")
        plane = _display(scan[:, :, index]).astype(np.float64)
    elif label_plane is not None:
        plane = np.zeros(label_plane.shape)
    else:
        plane = _display(data[:, :, index]).astype(np.float64)
    finite = plane[np.isfinite(plane)]
    if image_path is not None and finite.size and float(finite.min()) < -500:
        low, high = _CT_WINDOW
    elif finite.size:
        low, high = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    else:
        low, high = 0.0, 1.0
    gray = np.clip((plane - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((gray * 255)[:, :, None], 3, axis=2)
    if label_plane is not None:
        for value in np.unique(label_plane).tolist():
            if value == 0:
                continue
            selected = label_plane == value
            color = np.asarray(_PALETTE[int(value) % len(_PALETTE)], dtype=np.float64)
            rgb[selected] = 0.45 * rgb[selected] + 0.55 * color
    picture = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)))
    # Keep the physical aspect: columns follow x spacing, rows follow y spacing.
    x_mm, y_mm = (float(size) for size in volume.header.get_zooms()[:2])
    width, height = picture.width * x_mm, picture.height * y_mm
    scale = _PREVIEW_SIZE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    sharp = label_plane is not None and image_path is None
    picture = picture.resize(size, Image.Resampling.NEAREST if sharp else Image.Resampling.BILINEAR)
    legend = [
        (row["value"], line)
        for row in summary["labels"] if row["name"] is not None
        for line in textwrap.wrap(f"{row['value']}: {row['name']}", width=60)
    ]
    if legend:
        canvas = Image.new("RGB", (max(picture.width, 400), picture.height + 18 * len(legend) + 8))
        canvas.paste(picture, ((canvas.width - picture.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        for index, (value, line) in enumerate(legend):
            top = picture.height + 4 + index * 18
            draw.rectangle((4, top, 14, top + 10), fill=_PALETTE[value % len(_PALETTE)])
            draw.text((20, top), line, fill="white")
        picture = canvas
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("labels", json.dumps(summary["labels"]))
    metadata.add_text("names_from", names_from or "null")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


def _series(directory: Path) -> dict[str, list[str]]:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(_MISSING.format(name="SimpleITK", extra="dicom")) from error
    reader = sitk.ImageSeriesReader()
    return {
        uid: list(reader.GetGDCMSeriesFileNames(str(directory), uid))
        for uid in reader.GetGDCMSeriesIDs(str(directory))
    }


def _tag(file: str, tag: str) -> str | None:
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(file)
    reader.ReadImageInformation()
    if not reader.HasMetaDataKey(tag):
        return None
    return reader.GetMetaData(tag).strip() or None


def list_dicom_series(
    directory: str | Path, *, root: str | Path | None = None,
) -> list[DicomSeries]:
    """List the DICOM series in a folder (not its subfolders), with modality and description.

    Raises:
        NotADirectoryError: ``directory`` is not a folder.
        ImportError: SimpleITK is not installed.
    """
    folder = _resolve(directory, root)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return [
        {"series_uid": uid, "modality": _tag(files[0], "0008|0060"),
         "description": _tag(files[0], "0008|103e"), "files": len(files)}
        for uid, files in sorted(_series(folder).items())
    ]


def convert_dicom(
    directory: str | Path, output_path: str | Path, *, series_uid: str | None = None,
    root: str | Path | None = None,
) -> ConvertedSeries:
    """Write one DICOM series as NIfTI, keeping its geometry and rescaled values (HU for CT).

    With one series in the folder ``series_uid`` may be omitted. The output name
    must end in ``.nii.gz`` or ``.nii``; a completed temporary file is published
    without replacing an existing file, including one created during conversion.

    Raises:
        NotADirectoryError: ``directory`` is not a folder.
        ValueError: No series, several series without ``series_uid``, an unknown
            UID, a series that is not an image (such as SEG or RTSTRUCT), or a
            non-NIfTI output name.
        FileExistsError: ``output_path`` already exists.
        ImportError: SimpleITK is not installed.
    """
    folder = _resolve(directory, root)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    target = _resolve(output_path, root)
    if not target.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("output_path must end in .nii or .nii.gz")
    if target.exists():
        raise FileExistsError(target)
    series = _series(folder)
    if not series:
        raise ValueError(f"No DICOM series in {folder}")
    if series_uid is None:
        if len(series) > 1:
            raise ValueError(f"{len(series)} series in {folder}; choose one series_uid: "
                             + ", ".join(sorted(series)))
        series_uid = next(iter(series))
    if series_uid not in series:
        raise ValueError(f"Unknown series_uid {series_uid}")
    import SimpleITK as sitk

    files = series[series_uid]
    modality = _tag(files[0], "0008|0060")
    if modality in _NOT_IMAGES:
        raise ValueError(f"Series {series_uid} is a {modality} object, not an image; "
                         "choose an image series from list_dicom_series")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".medmlx-dicom-", dir=target.parent) as staging:
        temporary = Path(staging) / target.name
        sitk.WriteImage(image, str(temporary))
        os.link(temporary, target)
    return {
        "output_path": target.as_posix(), "series_uid": series_uid,
        "modality": modality, "description": _tag(files[0], "0008|103e"),
        "shape": [int(size) for size in image.GetSize()],
        "spacing_mm": [float(size) for size in image.GetSpacing()],
    }


__all__ = ["convert_dicom", "list_dicom_series", "preview_output", "summarize_output"]
