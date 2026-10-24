"""Channel-last MLX layers shared by standalone medical imaging models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.typing import MlxRuntime
from medmlx_core.upsample import deconv2x_ncdhw


class ChannelLastGraph:
    """Layer primitives over a Torch-named weight mapping, with channel-last activations.

    Activations are ``(N, *spatial, C)``. Weights keep the Torch layout and
    names (``<name>.weight``, ``<name>.bias``, ``<name>.running_mean``, ...), and
    are rearranged inside each call, so converted checkpoints need no layout pass.
    Missing weight keys raise ``KeyError``, except optional ``<name>.bias``.
    """

    def __init__(self, weights: Mapping[str, Array], *, mx: MlxRuntime) -> None:
        self.w = weights
        self.mx = mx

    def conv(
        self, x: Array, name: str, *, padding: int = 0, stride: int = 1, groups: int = 1
    ) -> Array:
        """Convolve with ``<name>.weight`` (Torch ``[O, I, *K]``) and optional bias.

        The weight rank selects ``conv3d`` (rank 5) or ``conv2d`` (otherwise).
        ``padding`` and ``stride`` are one int applied to every spatial axis.
        """
        w = self.w[f"{name}.weight"]
        axes = (0, *range(2, w.ndim), 1)
        operation = self.mx.conv3d if w.ndim == 5 else self.mx.conv2d
        y = operation(x, self.mx.transpose(w, axes), stride=stride, padding=padding, groups=groups)
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def linear(self, x: Array, name: str) -> Array:
        """Apply a Torch ``Linear`` (weight ``[out, in]``) over the last axis."""
        y = x @ self.w[f"{name}.weight"].T
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    def layer_norm(self, x: Array, name: str | None = None) -> Array:
        """Normalize the last axis with eps 1e-5; no affine terms when ``name`` is None."""
        weight = None if name is None else self.w[f"{name}.weight"]
        bias = None if name is None else self.w[f"{name}.bias"]
        return self.mx.fast.layer_norm(x, weight, bias, 1e-5)

    def batch_norm(self, x: Array, name: str, *, eps: float = 1e-5) -> Array:
        """Inference-mode batch norm from stored running statistics on the last axis."""
        scale = self.w[f"{name}.weight"] * self.mx.rsqrt(self.w[f"{name}.running_var"] + eps)
        return (x - self.w[f"{name}.running_mean"]) * scale + self.w[f"{name}.bias"]

    def relu(self, x: Array) -> Array:
        return self.mx.maximum(x, 0)

    def swish(self, x: Array) -> Array:
        return x * self.mx.sigmoid(x)

    def instance_norm(self, x: Array) -> Array:
        """Normalize each sample and channel over the spatial axes (eps 1e-5, no affine)."""
        axes = tuple(range(1, x.ndim - 1))
        mean = self.mx.mean(x, axis=axes, keepdims=True)
        return (x - mean) * self.mx.rsqrt(self.mx.var(x, axis=axes, keepdims=True) + 1e-5)

    def residual3d(self, x: Array, name: str) -> Array:
        """Apply a MONAI ``UnetResBlock`` as two 3x3x3 convolutions with a skip connection.

        Each convolution is followed by instance norm, with LeakyReLU (slope 0.01)
        after the first and after the residual sum. The skip is projected by a
        1x1x1 convolution plus instance norm only when ``<name>.conv3.conv.weight``
        exists. Weights are read from ``<name>.conv{1,2,3}.conv``.
        """
        def activate(y: Array) -> Array:
            return self.mx.where(y >= 0, y, y * 0.01)

        y = activate(self.instance_norm(self.conv(x, f"{name}.conv1.conv", padding=1)))
        y = self.instance_norm(self.conv(y, f"{name}.conv2.conv", padding=1))
        if f"{name}.conv3.conv.weight" in self.w:
            x = self.instance_norm(self.conv(x, f"{name}.conv3.conv"))
        return activate(y + x)

    def deconv3d(self, x: Array, name: str) -> Array:
        """Upsample 2x per spatial axis with ``<name>.weight`` (Torch ``[I, O, 2, 2, 2]``).

        The transposed convolution has no bias here; ``<name>.bias`` is ignored.
        """
        result = deconv2x_ncdhw(
            self.mx.transpose(x, (0, 4, 1, 2, 3)), self.w[f"{name}.weight"], None, mx=self.mx
        )
        return self.mx.transpose(result, (0, 2, 3, 4, 1))

    def require_input(self, x: Array, shape: tuple[int, ...]) -> None:
        """Require a float32 batch of ``shape`` (batch axis excluded), with batch >= 1.

        Raises:
            ValueError: The rank, trailing shape, batch size or dtype is wrong.
        """
        if x.ndim != len(shape) + 1 or tuple(x.shape[1:]) != shape or x.shape[0] < 1:
            raise ValueError(f"Expected Bx{'x'.join(map(str, shape))} input; got {x.shape}")
        if x.dtype != self.mx.float32:
            raise ValueError("MedMLX channel-last graphs require float32 inputs")
