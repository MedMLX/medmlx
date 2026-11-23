"""Transpose-convolution parity when a lazy decoder consumes the result."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numerical import assert_reference
from numpy.typing import NDArray

from medmlx_core.ops import conv_transpose3d_ncdhw
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime


def test_lazy_transpose_decoder_with_skip_matches_torch_repeated() -> None:
    pytest.importorskip("torch")
    import torch

    from_numpy = cast(Callable[[NDArray[np.float32]], torch.Tensor], vars(torch)["from_numpy"])
    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(43)
    values = rng.normal(size=(1, 32, 48, 48, 48)).astype(np.float32)
    weight = rng.normal(scale=0.1, size=(32, 32, 2, 2, 2)).astype(np.float32)
    bias = rng.normal(size=32).astype(np.float32)
    skip = rng.normal(size=(1, 32, 96, 96, 96)).astype(np.float32)
    expected = (
        torch.nn.functional.conv_transpose3d(
            from_numpy(values), from_numpy(weight), from_numpy(bias), stride=2
        ).numpy()
        + skip
    )

    # Small isolated outputs miss the upstream cross-encoder buffer hazard.
    # Keep the decoder addition lazy and repeat after allocator reuse.
    for _ in range(3):
        inputs = [mx.array(array) for array in (values, weight, bias, skip)]
        mx.eval(inputs)
        actual = (
            conv_transpose3d_ncdhw(*inputs[:3], padding=0, output_padding=0, stride=2, mx=mx)
            + inputs[3]
        )
        mx.eval(actual)
        assert_reference(cast(NDArray[np.float32], np.asarray(actual)), expected, budget="dot")
