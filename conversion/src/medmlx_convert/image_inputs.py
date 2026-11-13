"""Sourced CT and radiograph inputs for numerical conversion checks."""

import os
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from medmlx_convert.boundary import as_tensor, function


def ct_input(
    shape: tuple[int, ...] = (1, 1, 8, 8, 8),
    *, hu_window: tuple[float, float] | None = None,
) -> NDArray[np.float32]:
    """Return HCC-TACE pixels in NCDHW layout, cropped or nearest-sampled.

    Values use 1024-HU units unless a clipping window requests [0, 1]. Larger
    grids sample the available acquisition; channels and batches use reflected
    views of that acquisition, rather than representing independent modalities.
    """
    if len(shape) != 5 or any(size < 1 for size in shape):
        raise ValueError(f"Expected a positive NCDHW shape, got {shape}")
    cache = Path.home() / ".cache/medmlx/test-data"
    source = nib.Nifti1Image.from_filename(os.environ.get(
        "MEDMLX_TEST_CT", str(cache / "hcc_063_ct.nii.gz"),
    ))
    volume = np.asarray(source.dataobj, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT, got {volume.shape}")
    spatial = shape[2:]
    starts = tuple(min(origin, max(size - wanted, 0)) for origin, size, wanted
                   in zip((100, 210, 55), volume.shape, spatial, strict=True))
    crop = volume[tuple(slice(start, start + wanted)
                        for start, wanted in zip(starts, spatial, strict=True))]
    indices = [np.arange(wanted) * size // wanted
               for size, wanted in zip(crop.shape, spatial, strict=True)]
    values = crop[np.ix_(*indices)]
    if hu_window is None:
        values = values / np.float32(1024)
    else:
        lower, upper = hu_window
        if upper <= lower:
            raise ValueError("HU window must have increasing bounds")
        values = np.clip((values - np.float32(lower)) / np.float32(upper - lower), 0, 1)
    return np.stack([np.flip(values, axis=index % 3) if index else values
                     for index in range(shape[0] * shape[1])]).reshape(shape).copy()


def chest_input(
    size: tuple[int, int], *, normalized: bool = True, channels: Literal[1, 3] = 3,
) -> NDArray[np.float32]:
    """Resize the radiograph to (width, height), returning NCHW FP32 pixels.

    ``normalized`` selects [-1, 1]; otherwise values remain in [0, 1].
    """
    cache = Path.home() / ".cache/medmlx/test-data"
    path = os.environ.get("MEDMLX_TEST_XRAY", str(cache / "chest_xray.jpg"))
    with Image.open(path) as source:
        values = np.asarray(source.convert("L" if channels == 1 else "RGB").resize(size),
                            dtype=np.float32)
    if channels == 1:
        values = values[..., None]
    result = values.transpose(2, 0, 1)[None] / np.float32(255)
    if normalized:
        result = result * np.float32(2) - np.float32(1)
    return result.copy()


def ct_tensor(
    shape: tuple[int, ...], *, hu_window: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Expose the CT array at the upstream Torch network boundary."""
    return as_tensor(function(torch, "from_numpy")(ct_input(shape, hu_window=hu_window)))


def chest_tensor(
    size: tuple[int, int], *, normalized: bool = True, channels: Literal[1, 3] = 3,
) -> torch.Tensor:
    """Expose the resized radiograph at the upstream Torch network boundary."""
    return as_tensor(function(torch, "from_numpy")(
        chest_input(size, normalized=normalized, channels=channels),
    ))
