"""Checkpoint-free MLX stride-2 ConvTranspose3d cross-encoder reproducer."""

from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


def main() -> None:
    mx.set_default_device(mx.gpu)
    evaluate = cast(Callable[..., None], vars(mx)["eval"])
    rng = np.random.default_rng(43)
    values = rng.normal(size=(1, 48, 48, 48, 32)).astype(np.float32)
    weight = rng.normal(scale=0.1, size=(32, 2, 2, 2, 32)).astype(np.float32)
    bias = rng.normal(size=32).astype(np.float32)
    skip = rng.normal(size=(1, 96, 96, 96, 32)).astype(np.float32)

    expected = np.empty_like(skip)
    for depth in range(2):
        for height in range(2):
            for width in range(2):
                expected[:, depth::2, height::2, width::2] = (
                    values @ weight[:, depth, height, width].T + bias
                )
    expected += skip

    print(f"MLX {version('mlx')}; FP32; NDHWC input {values.shape}")
    # The middle call isolates the operation; the final lazy call exercises
    # allocator reuse after command-buffer completion.
    for evaluate_deconv in (False, True, False):
        x, w, b, residual = (mx.array(array) for array in (values, weight, bias, skip))
        evaluate(x, w, b, residual)
        decoded = mx.conv_transpose3d(x, w, stride=2)
        if evaluate_deconv:
            evaluate(decoded)
        output = decoded + b + residual
        evaluate(output)
        host = cast(NDArray[np.float32], np.asarray(output))
        difference = host - expected
        relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(expected))
        mode = "evaluated deconv" if evaluate_deconv else "lazy consumer"
        print(f"{mode}: relative L2={relative_l2:.8g}, max abs={np.abs(difference).max():.8g}")


if __name__ == "__main__":
    main()
