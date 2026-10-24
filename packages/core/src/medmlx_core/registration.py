"""Bounded SynthMorph geometry adapters, independent of learned model conversion.

Coordinates are zero-based ij voxel indices, and displacements are pull fields:
output[x] samples input[x + displacement[x]]. No affine matrices, centered grids,
world coordinates or channelwise transforms are accepted here. The contract
matches VoxelMorph transform and stationary scaling-and-squaring with Neurite
linear interpolation; see the pinned oracle fixture generator.
"""

import math
from itertools import product

import mlx.core as mx


def _volume(value: mx.array, channels: int | None = None) -> None:
    if (value.ndim != 4 or value.dtype != mx.float32 or min(value.shape) < 1
            or (channels is not None and value.shape[-1] != channels)):
        raise ValueError("Registration requires nonempty float32 DHWC tensors")


def dense_warp3d(volume: mx.array, displacement: mx.array, *,
                 fill_value: float | None = None) -> mx.array:
    """Resample DHWC at a DHW3 voxel pull field with trilinear interpolation.

    None extends the nearest border value. A finite fill constant replaces the
    entire sample whenever any un-clipped coordinate is outside the input grid.
    The output spatial shape comes from the displacement, and may differ from
    the input. Model arithmetic is explicitly scheduled on MLX Metal.

    Args:
        volume: Float32 ``(D, H, W, C)`` array to sample.
        displacement: Float32 ``(D', H', W', 3)`` pull field in voxel units, with
            components ordered like the volume's D, H, W axes.
        fill_value: Finite constant for out-of-grid samples, or None to extend the
            nearest border value.

    Returns:
        Float32 ``(D', H', W', C)`` array.

    Raises:
        ValueError: An input is not a nonempty float32 rank-4 array, the
            displacement does not have 3 channels, or ``fill_value`` is not finite.
        RuntimeError: Metal is unavailable; there is no CPU fallback.
    """
    _volume(volume)
    _volume(displacement, 3)
    if not mx.metal.is_available():
        raise RuntimeError("Registration requires Metal; no CPU fallback")
    if fill_value is not None and not math.isfinite(fill_value):
        raise ValueError("Registration fill value must be finite")
    with mx.stream(mx.gpu):
        grid = mx.meshgrid(*(mx.arange(size, dtype=mx.float32)
                             for size in displacement.shape[:3]), indexing="ij")
        location = [grid[dim] + displacement[..., dim] for dim in range(3)]
        clipped = [mx.clip(value, 0, volume.shape[dim] - 1)
                   for dim, value in enumerate(location)]
        lower = [mx.floor(value) for value in clipped]
        upper = [mx.minimum(value + 1, volume.shape[dim] - 1)
                 for dim, value in enumerate(lower)]
        lower_weight = [upper[d] - clipped[d] for d in range(3)]
        upper_weight = [1 - value for value in lower_weight]
        indices = [tuple(value.astype(mx.int32) for value in lower),
                   tuple(value.astype(mx.int32) for value in upper)]
        weights = [lower_weight, upper_weight]
        result = mx.zeros((*displacement.shape[:3], volume.shape[-1]), dtype=mx.float32)
        # Eight corners, vectorized across every voxel; preserve source sum order.
        for corner in product((0, 1), repeat=3):
            index = tuple(indices[corner[d]][d] for d in range(3))
            weight = weights[corner[0]][0] * weights[corner[1]][1] * weights[corner[2]][2]
            result = result + weight[..., None] * volume[index]
        if fill_value is not None:
            outside = mx.zeros(displacement.shape[:3], dtype=mx.bool_)
            for dim, value in enumerate(location):
                outside = outside | (value < 0) | (value > volume.shape[dim] - 1)
            result = mx.where(outside[..., None], fill_value, result)
        return result


def integrate_stationary_velocity(velocity: mx.array, *, steps: int = 7) -> mx.array:
    """Integrate a DHW3 stationary voxel velocity by the source scaling-and-squaring rule.

    This implements vec / 2**steps followed by vec += warp(vec, vec), using border
    extension. It does not support time-dependent fields or ODE integration.

    Args:
        velocity: Float32 ``(D, H, W, 3)`` stationary velocity in voxel units.
        steps: Number of squaring steps; 0 only rescales by 1 and returns the
            input values.

    Returns:
        Float32 ``(D, H, W, 3)`` displacement field.

    Raises:
        ValueError: ``velocity`` is not a nonempty float32 ``(D, H, W, 3)`` array,
            or ``steps`` is not a nonnegative ``int``.
        RuntimeError: Metal is unavailable; there is no CPU fallback.
    """
    _volume(velocity, 3)
    if type(steps) is not int or steps < 0:
        raise ValueError("Scaling-and-squaring steps must be a nonnegative integer")
    if not mx.metal.is_available():
        raise RuntimeError("Registration requires Metal; no CPU fallback")
    with mx.stream(mx.gpu):
        displacement = velocity / (2 ** steps)
        for _ in range(steps):
            displacement = displacement + dense_warp3d(displacement, displacement)
        return displacement
