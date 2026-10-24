"""Typed MLX boundaries without importing the optional runtime at package import."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import Device, DeviceType, Dtype, StreamOrDevice, array

type HostArray = NDArray[np.generic]
type Scalar = int | float | bool


class ArrayFactory(Protocol):
    def __call__(
        self, val: array | HostArray | list[float], dtype: Dtype | None = None
    ) -> array: ...


class ArrayEvaluator(Protocol):
    def __call__(self, *args: array | Sequence[array]) -> None: ...


class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[array],
        template: list[tuple[str, int | bool]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Dtype],
        stream: StreamOrDevice = None,
    ) -> list[array]: ...


class MetalRuntime(Protocol):
    def is_available(self) -> bool: ...


class FastRuntime(Protocol):
    def layer_norm(
        self,
        x: array,
        weight: array | None,
        bias: array | None,
        eps: float,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def metal_kernel(
        self,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
        compile_options: object | None = None,
    ) -> object: ...


class MlxRuntime(Protocol):
    @property
    def array(self) -> type[array]: ...
    @property
    def float32(self) -> Dtype: ...
    @property
    def float16(self) -> Dtype: ...
    @property
    def int32(self) -> Dtype: ...
    @property
    def gpu(self) -> DeviceType: ...
    @property
    def fast(self) -> FastRuntime: ...
    @property
    def metal(self) -> MetalRuntime: ...
    def eval(self, *args: array | Sequence[array]) -> None: ...
    def reset_peak_memory(self) -> None: ...
    def get_peak_memory(self) -> int: ...
    def default_device(self) -> Device: ...
    def set_default_device(self, device: Device | DeviceType) -> None: ...
    def reshape(
        self, a: array, /, shape: Sequence[int], *, stream: StreamOrDevice = None
    ) -> array: ...
    def rsqrt(self, a: array, /, *, stream: StreamOrDevice = None) -> array: ...
    def sigmoid(self, a: array, /, *, stream: StreamOrDevice = None) -> array: ...
    @overload
    def arange(
        self,
        start: int | float,
        stop: int | float | None,
        step: int | float | None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    @overload
    def arange(
        self,
        stop: int | float,
        step: int | float | None = None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def take(
        self,
        a: array,
        /,
        indices: int | array,
        axis: int | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def zeros(
        self,
        shape: int | Sequence[int],
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def minimum(
        self, a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array: ...
    def maximum(
        self, a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array: ...
    def floor(self, a: array, /, *, stream: StreamOrDevice = None) -> array: ...
    def swapaxes(
        self, a: array, /, axis1: int, axis2: int, *, stream: StreamOrDevice = None
    ) -> array: ...
    def transpose(
        self, a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None
    ) -> array: ...
    def mean(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def var(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        ddof: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def concatenate(
        self, arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array: ...
    def stack(
        self, arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array: ...
    def repeat(
        self, array: array, repeats: int, axis: int | None = None, *, stream: StreamOrDevice = None
    ) -> array: ...
    def pad(
        self,
        a: array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        constant_values: Scalar | array = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def conv2d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def conv3d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def conv_transpose3d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
    def where(
        self,
        condition: Scalar | array,
        x: Scalar | array,
        y: Scalar | array,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
