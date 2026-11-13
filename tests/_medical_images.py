"""HCC-TACE CT pixels for controlled workflow and output test grids."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def ct_crop(shape: tuple[int, int, int]) -> NDArray[np.float32]:
    """Read source HU values without changing their order or intensity scale."""
    path = Path(os.environ.get(
        "MEDMLX_TEST_CT", str(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"),
    ))
    load = cast(Callable[[Path], object], vars(nib)["load"])
    source = load(path)
    if not isinstance(source, nib.Nifti1Image):
        raise ValueError("The CT fixture must be a NIfTI image")
    data = np.asarray(source.dataobj, dtype=np.float32)
    crop = data[100:100 + shape[0], 210:210 + shape[1], 55:55 + shape[2]].copy()
    if crop.shape != shape:
        raise ValueError("The HCC-TACE CT does not cover the requested crop")
    return crop
