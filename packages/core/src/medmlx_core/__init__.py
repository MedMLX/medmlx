"""Shared MLX runtime for MedMLX model packages."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medmlx_core.sliding_window import (
        compute_importance_map,
        dense_patch_slices,
        sliding_window_inference,
    )

from medmlx_core.checkpoints import (
    load_torch_checkpoint,
    mapping_from_pairs,
    tensor_mapping_from_payload,
)
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MedmlxError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.layout import (
    conv3d_weight_to_mlx,
    conv_transpose3d_weight_to_mlx,
    require_ncdhw,
    to_ncdhw,
    to_ndhwc,
    tokens_from_ncdhw,
    tokens_to_ncdhw,
)
from medmlx_core.ops import (
    as_fp32,
    avg_pool3d_ncdhw,
    concat_channels_ncdhw,
    conv3d_ncdhw,
    conv_transpose3d_ncdhw,
    group_norm_ncdhw,
    linear,
    pad_spatial_trailing_ncdhw,
    silu,
    split_conv3d_ncdhw,
    split_conv_transpose3d_ncdhw,
    upsample_nearest_ncdhw,
    upsample_trilinear_ncdhw,
)
from medmlx_core.precision import Float32Operators
from medmlx_core.runtime import (
    MLX_EXTRA,
    TF32_VARIABLE,
    MlxHostReport,
    default_full_float32,
    import_mlx,
    mlx_default_device_name,
    probe_mlx_runtime,
    require_full_float32,
    require_mlx_device,
    require_mlx_runtime,
)
from medmlx_core.upsample import deconv2x_ncdhw, upsample_add_ncdhw
from medmlx_core.weights import DownloadProgress, weights_dir, weights_revision

# Set the default before this package runs MLX operations. An earlier MLX
# operation elsewhere in the process may already have fixed its setting.
default_full_float32()

__version__ = "0.2.5"

__all__ = [
    "MLX_EXTRA",
    "TF32_VARIABLE",
    "AssetNotReadyError",
    "DownloadProgress",
    "Float32Operators",
    "InvalidInputError",
    "MedmlxError",
    "MissingDependencyError",
    "MlxHostReport",
    "ModelExecutionError",
    "as_fp32",
    "avg_pool3d_ncdhw",
    "compute_importance_map",
    "concat_channels_ncdhw",
    "conv3d_ncdhw",
    "conv3d_weight_to_mlx",
    "conv_transpose3d_ncdhw",
    "conv_transpose3d_weight_to_mlx",
    "deconv2x_ncdhw",
    "default_full_float32",
    "dense_patch_slices",
    "group_norm_ncdhw",
    "import_mlx",
    "linear",
    "load_torch_checkpoint",
    "mapping_from_pairs",
    "mlx_default_device_name",
    "pad_spatial_trailing_ncdhw",
    "probe_mlx_runtime",
    "require_full_float32",
    "require_mlx_device",
    "require_mlx_runtime",
    "require_ncdhw",
    "silu",
    "sliding_window_inference",
    "split_conv3d_ncdhw",
    "split_conv_transpose3d_ncdhw",
    "tensor_mapping_from_payload",
    "to_ncdhw",
    "to_ndhwc",
    "tokens_from_ncdhw",
    "tokens_to_ncdhw",
    "upsample_add_ncdhw",
    "upsample_nearest_ncdhw",
    "upsample_trilinear_ncdhw",
    "weights_dir",
    "weights_revision",
]

# The sliding window imports MLX at module level, so load it on first use to
# keep `import medmlx_core` from importing MLX.
_LAZY = {
    "compute_importance_map": "medmlx_core.sliding_window",
    "dense_patch_slices": "medmlx_core.sliding_window",
    "sliding_window_inference": "medmlx_core.sliding_window",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
