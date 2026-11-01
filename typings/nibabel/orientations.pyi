"""Installed NiBabel orientation operations used by CT mask restoration."""

import numpy as np
from numpy.typing import NDArray

def io_orientation(affine: NDArray[np.float64]) -> NDArray[np.float64]: ...
def ornt_transform(
    start_ornt: NDArray[np.float64],
    end_ornt: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def apply_orientation(
    arr: NDArray[np.float32],
    ornt: NDArray[np.float64],
) -> NDArray[np.float32]: ...
