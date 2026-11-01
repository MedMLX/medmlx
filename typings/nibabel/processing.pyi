"""Consumed NiBabel 5.4 resampling signatures; interpolation runs on the host."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nibabel import Nifti1Image

def resample_from_to(
    from_img: Nifti1Image,
    to_vox_map: Nifti1Image | tuple[tuple[int, ...], NDArray[np.float64]],
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    out_class: type[Nifti1Image] = Nifti1Image,
) -> Nifti1Image: ...
def resample_to_output(
    in_img: Nifti1Image,
    voxel_sizes: Sequence[float] | NDArray[np.float64] | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    out_class: type[Nifti1Image] = Nifti1Image,
) -> Nifti1Image: ...
