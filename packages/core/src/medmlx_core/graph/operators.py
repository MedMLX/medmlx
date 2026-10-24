"""Metal execution of the portable inference operations declared by core.

Source tensors remain NCHW/NCDHW between operations. Convolution and pooling
perform explicit local channel-last adaptations. Float32 accumulation may differ
from Torch, so whole-network differential checks remain mandatory.
"""

from __future__ import annotations

from importlib import import_module
from typing import cast

import mlx.core as mx

from medmlx_core.graph._mlx import invoke, tensor
from medmlx_core.graph.schema import Op, Scalar

type Result = mx.array | Scalar | tuple[Result, ...]
type Parameters = dict[str, Result]


def array(p: Parameters, name: str) -> mx.array:
    value = p[name]
    if not isinstance(value, mx.array):
        raise ValueError(f"{name} must be a tensor")
    return value


def ints(value: Result) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(v) is not int for v in value):
        raise ValueError(f"Expected integer tuple: {value}")
    return cast(tuple[int, ...], value)


def _last(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, *range(2, x.ndim), 1))


def _first(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, x.ndim - 1, *range(1, x.ndim - 1)))


def _conv(op: Op, p: Parameters) -> mx.array:
    x, weight = array(p, "input"), array(p, "weight")
    transpose = op in (Op.DECONV2, Op.DECONV3)
    groups = cast(int, p["groups"])
    if groups != 1 and (x.ndim == 5 or transpose):
        # MLX's conv3d, conv_transpose2d and conv_transpose3d all document
        # "Only the default groups=1 is currently supported" as of mlx 0.32.4,
        # the pinned floor; only non-transposed 2D convolution accepts groups.
        raise ValueError("MLX only supports groups=1 for 3D or transposed convolutions")
    if (op == Op.DECONV3 and tuple(weight.shape[2:]) == (2, 2, 2)
            and ints(p["stride"]) == (2, 2, 2) and not any(ints(p["padding"]))
            and not any(ints(p["output_padding"])) and ints(p["dilation"]) == (1, 1, 1)):
        # The shared one-matmul implementation avoids MLX's lazy phase-view
        # synchronization defect in the generic 2x decoder path, confirmed
        # still reproducible on the pinned mlx>=0.32.4 floor.
        core = import_module("medmlx_core.upsample")
        return tensor(invoke(core, "deconv2x_ncdhw", x, weight, p["bias"], mx=mx))
    axes = (1, *range(2, weight.ndim), 0) if transpose else (
        0, *range(2, weight.ndim), 1)
    weight = mx.transpose(weight, axes)
    kwargs = {name: ints(p[name]) for name in ("stride", "padding", "dilation")}
    conv = {Op.CONV2: "conv2d", Op.CONV3: "conv3d",
            Op.DECONV2: "conv_transpose2d", Op.DECONV3: "conv_transpose3d"}[op]
    if transpose:
        kwargs["output_padding"] = ints(p["output_padding"])
    result = tensor(invoke(mx, conv, _last(x), weight, groups=groups, **kwargs))
    if p["bias"] is not None:
        result = result + array(p, "bias")
    return _first(result)


def _norm(op: Op, p: Parameters) -> mx.array:
    x = array(p, "input")
    broadcast = (1, x.shape[1], *((1,) * (x.ndim - 2)))
    if op != Op.INSTANCE_NORM:
        if op == Op.BATCH_NORM and p["training"] is not False:
            raise ValueError("Training BatchNorm is unsupported")
        mean = mx.reshape(array(p, "running_mean"), broadcast)
        variance = mx.reshape(array(p, "running_var"), broadcast)
    elif p["use_input_stats"] is True:
        axes = tuple(range(2, x.ndim))
        mean = mx.mean(x, axis=axes, keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=axes, keepdims=True)
    else:
        mean = mx.reshape(array(p, "running_mean"), broadcast)
        variance = mx.reshape(array(p, "running_var"), broadcast)
    result = (x - mean) * mx.rsqrt(variance + cast(float, p["eps"]))
    if p["weight"] is not None:
        result = result * mx.reshape(array(p, "weight"), broadcast)
    if p["bias"] is not None:
        result = result + mx.reshape(array(p, "bias"), broadcast)
    return result


def _pool(p: Parameters) -> mx.array:
    padding = p["padding"]
    padded = any(ints(padding)) if isinstance(padding, tuple) else padding != 0
    if (p["ceil_mode"] is not False or p["divisor_override"] is not None or padded):
        raise ValueError("Pilot AvgPool2d requires floor mode, zero padding and default divisor")
    kernel = ints(p["kernel_size"])
    stride = ints(p["stride"]) or kernel
    pool = invoke(import_module("mlx.nn"), "AvgPool2d", kernel_size=kernel, stride=stride)
    return _first(tensor(invoke(pool, "__call__", _last(array(p, "input")))))


def _bilinear(p: Parameters) -> mx.array:
    x = array(p, "input")
    scales = p["scale_factors"]
    if scales is None:
        sizes = ints(p["output_size"])
        factors = tuple(size / old for size, old in zip(sizes, x.shape[2:], strict=True))
    else:
        factors = cast(tuple[float, ...], scales)
    if any(factor != int(factor) or factor < 1 for factor in factors):
        raise ValueError("Pilot bilinear requires positive integral upsampling factors")
    up = invoke(import_module("mlx.nn"), "Upsample", scale_factor=factors, mode="linear",
                align_corners=cast(bool, p["align_corners"]))
    return _first(tensor(invoke(up, "__call__", _last(x))))


def execute(op: Op, p: Parameters) -> Result:
    """Execute one typed operation or raise for a variant outside the pilot contract."""
    if op in (Op.CONV2, Op.CONV3, Op.DECONV2, Op.DECONV3):
        return _conv(op, p)
    if op in (Op.BATCH_NORM, Op.INSTANCE_NORM):
        return _norm(op, p)
    if op == Op.EVAL_BATCH_NORM:
        # Inference auxiliary outputs have no saved training statistics.
        empty = mx.zeros((0,), dtype=mx.float32)
        return _norm(op, p), empty, empty
    if op == Op.AVG_POOL2:
        return _pool(p)
    if op == Op.BILINEAR:
        return _bilinear(p)
    return _elementary(op, p)


def _elementary(op: Op, p: Parameters) -> Result:
    if op == Op.ASSERT_META:
        x = array(p, "input")
        if p["shape"] is not None and tuple(x.shape) != ints(p["shape"]):
            raise ValueError("Captured tensor size assertion failed")
        if (p["dtype"] is not None
                and str(x.dtype).removeprefix("mlx.core.") != p["dtype"]):
            raise ValueError("Captured tensor dtype assertion failed")
        return None
    if op == Op.GETITEM:
        return cast(tuple[Result, ...], p["value"])[cast(int, p["index"])]
    if op == Op.CAT:
        return mx.concatenate(list(cast(tuple[mx.array, ...], p["tensors"])),
                              axis=cast(int, p["axis"]))
    x = array(p, "input")
    if op == Op.RELU:
        return mx.maximum(x, 0)
    if op == Op.LEAKY_RELU:
        return mx.where(x >= 0, x, x * cast(float, p["negative_slope"]))
    if op == Op.LOG_SOFTMAX:
        if p["dtype"] is not None:
            raise ValueError("LogSoftmax dtype override is outside the FP32 pilot")
        return x - mx.logsumexp(x, axis=cast(int, p["axis"]), keepdims=True)
    if op == Op.SLICE:
        slices = [slice(None)] * x.ndim
        dim = cast(int, p["axis"])
        source = slice(cast(int, p["start"]), cast(int, p["end"]), cast(int, p["step"]))
        # Normalize open/large bounds before passing them to MLX's 32-bit indices.
        slices[dim] = slice(*source.indices(x.shape[dim]))
        return x[tuple(slices)]
    if op == Op.RESHAPE:
        return mx.reshape(x, ints(p["shape"]))
    if op == Op.SPLIT:
        size, dim = cast(int, p["chunk_size"]), cast(int, p["axis"])
        if size <= 0:
            raise ValueError("Pilot split requires positive chunk size")
        return tuple(mx.split(x, list(range(size, x.shape[dim], size)), axis=dim))
    if op == Op.CAST:
        dtype = {"float32": mx.float32, "int32": mx.int32, "bool": mx.bool_}[str(p["dtype"])]
        return x.astype(dtype)
    other = cast(mx.array | float, p["other"])
    if op == Op.ADD:
        return x + other * cast(float, p["alpha"])
    if op == Op.SUB:
        return x - other * cast(float, p["alpha"])
    if op == Op.MUL:
        return x * other
    if op == Op.DIV:
        return x / other
    raise ValueError(f"No MLX implementation for {op}")
