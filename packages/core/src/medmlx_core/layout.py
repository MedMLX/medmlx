"""NCDHW module layout and NDHWC remaps for MLX 3D conv/norm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.typing import MlxRuntime

LAYOUT_NCDHW = "maisi.layout.ncdhw"
LAYOUT_TO_NDHWC = "maisi.layout.to_ndhwc"
LAYOUT_TO_NCDHW = "maisi.layout.to_ncdhw"
LAYOUT_CONV3D_W = "maisi.layout.conv3d_w"
LAYOUT_CONV_TRANSPOSE3D_W = "maisi.layout.conv_transpose3d_w"
LAYOUT_LINEAR_W = "maisi.layout.linear_w"
LAYOUT_EMBED_W = "maisi.layout.embed_w"
LAYOUT_GN_AFFINE = "maisi.layout.gn_affine"
LAYOUT_VECTOR = "maisi.layout.vector"
IDENTITY_WEIGHT_LAYOUTS = frozenset(
    {LAYOUT_LINEAR_W, LAYOUT_EMBED_W, LAYOUT_GN_AFFINE, LAYOUT_VECTOR}
)


class RankedArray(Protocol):
    """Anything with an ``ndim`` attribute, so layout checks avoid importing MLX."""

    @property
    def ndim(self) -> int: ...


def require_ncdhw(array: RankedArray, *, name: str) -> None:
    """Raise if *array* is not a rank-5 NCDHW tensor."""

    if int(array.ndim) != 5:
        raise ValueError(f"{name} must be NCDHW rank-5; got ndim={array.ndim}")


def to_ndhwc(array: Array, mx: MlxRuntime) -> Array:
    """Transpose NCDHW to NDHWC, the layout MLX conv and norm layers expect."""

    return mx.transpose(array, (0, 2, 3, 4, 1))


def to_ncdhw(array: Array, mx: MlxRuntime) -> Array:
    """Transpose NDHWC back to the NCDHW layout used between modules."""

    return mx.transpose(array, (0, 4, 1, 2, 3))


def conv3d_weight_to_mlx(weight: Array, mx: MlxRuntime) -> Array:
    """Transpose a Torch conv3d weight [O, I, KD, KH, KW] to MLX [O, KD, KH, KW, I].

    Raises:
        ValueError: ``weight`` is not rank 5.
    """

    if int(weight.ndim) != 5:
        raise ValueError(f"conv3d weight must be rank-5 [O,I,K]; got ndim={weight.ndim}")
    return mx.transpose(weight, (0, 2, 3, 4, 1))


def conv_transpose3d_weight_to_mlx(weight: Array, mx: MlxRuntime) -> Array:
    """Transpose a Torch conv_transpose3d weight [I, O, KD, KH, KW] to MLX [O, KD, KH, KW, I].

    Raises:
        ValueError: ``weight`` is not rank 5.
    """

    if int(weight.ndim) != 5:
        raise ValueError(f"conv_transpose3d weight must be rank-5 [I,O,K]; got ndim={weight.ndim}")
    return mx.transpose(weight, (1, 2, 3, 4, 0))


def tokens_from_ncdhw(array: Array, mx: MlxRuntime) -> Array:
    """Flatten NCDHW [B,C,D,H,W] to tokens [B, D*H*W, C], depth-major then height, width.

    Raises:
        ValueError: ``array`` is not rank 5.
    """

    require_ncdhw(array, name="attention input")
    batch, channels, depth, height, width = (int(dim) for dim in array.shape)
    flat = mx.reshape(array, (batch, channels, depth * height * width))
    return mx.transpose(flat, (0, 2, 1))


def tokens_to_ncdhw(
    tokens: Array, spatial: tuple[int, int, int, int, int], mx: MlxRuntime
) -> Array:
    """Inverse of ``tokens_from_ncdhw``: tokens [B, N, C] to NCDHW.

    Args:
        tokens: Array of shape [B, D*H*W, C].
        spatial: The full target shape ``(B, C, D, H, W)``, not just the spatial
            extent; its element count must equal ``tokens.size``.
        mx: MLX module.

    The reshape fails inside MLX if ``spatial`` does not match the token count;
    this function does not check it first.
    """

    batch, channels, depth, height, width = spatial
    ncdhw = mx.transpose(tokens, (0, 2, 1))
    return mx.reshape(ncdhw, (batch, channels, depth, height, width))
