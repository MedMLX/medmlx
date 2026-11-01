"""NiBabel 5.4 public affine helpers used by CT spatial grounding."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def apply_affine(
    aff: NDArray[np.float64],
    pts: NDArray[np.generic] | Sequence[float],
    inplace: bool = False,
) -> NDArray[np.float64]: ...
def voxel_sizes(affine: NDArray[np.float64]) -> NDArray[np.float64]: ...
