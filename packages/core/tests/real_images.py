"""Real CT and X-ray inputs for shared image operations."""

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from PIL import Image


@lru_cache(maxsize=1)
def _ct() -> npt.NDArray[np.float32]:
    path = Path(
        os.environ.get(
            "MEDMLX_TEST_CT", str(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz")
        )
    )
    load = cast(Callable[[Path], object], vars(nib)["load"])
    image = load(path)
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError("The CT fixture must be a NIfTI image")
    return np.asarray(image.dataobj, dtype=np.float32)


@lru_cache(maxsize=1)
def _xray() -> npt.NDArray[np.float32]:
    path = Path(
        os.environ.get(
            "MEDMLX_TEST_XRAY", str(Path.home() / ".cache/medmlx/test-data/chest_xray.jpg")
        )
    )
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.float32)


def real_image(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    """Return normalized real-image crops in batch, channel, spatial order.

    Large 3D requests sample the original CT grid by nearest neighbor when the
    requested depth exceeds its 99 slices. Flips distinguish batches/channels.
    Intensities are quantized to 1/32 for the exact window-blending contracts.
    """
    spatial = shape[2:]
    if len(spatial) == 3:
        source = _ct()
        starts = tuple(
            (size - min(size, wanted)) // 2
            for size, wanted in zip(source.shape, spatial, strict=True)
        )
        crop = source[
            tuple(
                slice(start, start + min(size, wanted))
                for start, size, wanted in zip(starts, source.shape, spatial, strict=True)
            )
        ]
        indices = [
            np.arange(wanted) * size // wanted
            for size, wanted in zip(crop.shape, spatial, strict=True)
        ]
        image = crop[np.ix_(*indices)] / np.float32(1024)
    elif len(spatial) == 2:
        source = _xray()
        starts = tuple(
            (size - wanted) // 2 for size, wanted in zip(source.shape, spatial, strict=True)
        )
        if any(start < 0 for start in starts):
            raise ValueError(f"X-ray is too small for crop {spatial}")
        image = source[
            tuple(
                slice(start, start + wanted) for start, wanted in zip(starts, spatial, strict=True)
            )
        ] / np.float32(256)
    else:
        raise ValueError(f"Expected two or three spatial dimensions, got {spatial}")
    image = np.rint(image * np.float32(32)) / np.float32(32)
    return (
        np.asarray(np.stack(
            [
                np.flip(image, axis=i % len(spatial)) if i else image
                for i in range(shape[0] * shape[1])
            ]
        ), dtype=np.float32)
        .reshape(shape)
        .copy()
    )
