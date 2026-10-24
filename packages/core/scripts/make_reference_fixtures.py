#!/usr/bin/env python3
"""Record seeded helper outputs from pinned MONAI/PyTorch and NumPy float64."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiConvolution
from reference_support import (
    ArrayCase,
    CaseRecord,
    HostArray,
    array_cases,
    output_directory,
    save_fixture,
)
from torch.nn import functional as F

from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])


def layout_reference(case: ArrayCase, args: list[HostArray | None]) -> HostArray | None:
    source = args[0]
    assert source is not None, "Layout cases require an array"
    name, x = case.function, source.astype(np.float64)
    axes = {
        "to_ndhwc": (0, 2, 3, 4, 1),
        "to_ncdhw": (0, 4, 1, 2, 3),
        "conv3d_weight_to_mlx": (0, 2, 3, 4, 1),
        "conv_transpose3d_weight_to_mlx": (1, 2, 3, 4, 0),
    }
    if name == "require_ncdhw":
        return None
    if name in axes:
        return x.transpose(axes[name]).astype(np.float32)
    if name == "tokens_from_ncdhw":
        return x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1).astype(np.float32)
    if name == "tokens_to_ncdhw":
        options = case.options
        assert "spatial" in options
        return x.transpose(0, 2, 1).reshape(options["spatial"]).astype(np.float32)
    raise ValueError(f"No layout reference for {name}")


def split_reference(case: ArrayCase, args: list[torch.Tensor | None]) -> torch.Tensor:
    x, weight, bias = args
    assert x is not None and weight is not None, "Convolutions require input and weights"
    options = case.options.copy()
    transposed = "transpose" in case.function
    layer = MaisiConvolution(
        spatial_dims=3,
        in_channels=x.shape[1],
        out_channels=weight.shape[1] if transposed else weight.shape[0],
        num_splits=options.pop("num_splits", 1),
        dim_split=options.pop("dim_split", 1),
        print_info=False,
        save_mem=False,
        strides=options.pop("stride", 1),
        kernel_size=weight.shape[2:],
        conv_only=True,
        is_transposed=transposed,
        bias=bias is not None,
        padding=options.get("padding", 0),
        output_padding=options.get("output_padding", 0),
    )
    state = {"conv.conv.weight": weight}
    if bias is not None:
        state["conv.conv.bias"] = bias
    layer.load_state_dict(state, strict=True)
    return cast(torch.Tensor, layer(x))


def ops_reference(case: ArrayCase, args: list[torch.Tensor | None]) -> torch.Tensor:
    name, options = case.function, case.options
    x = args[0]
    assert x is not None, "Ops cases require an input tensor"
    if name.startswith("split_conv"):
        return split_reference(case, args)
    if name == "as_fp32":
        return x.float()
    if name == "silu":
        return F.silu(x)
    if name in ("linear", "conv3d_ncdhw", "conv_transpose3d_ncdhw"):
        weight = args[1]
        assert weight is not None, "Linear and convolution cases require weights"
        if name == "linear":
            return F.linear(x, weight, args[2])
    if name == "conv3d_ncdhw":
        assert "padding" in options
        weight = args[1]
        assert weight is not None
        return F.conv3d(
            x, weight, args[2], stride=options.get("stride", 1), padding=options["padding"]
        )
    if name == "conv_transpose3d_ncdhw":
        assert "padding" in options and "output_padding" in options
        weight = args[1]
        assert weight is not None
        return F.conv_transpose3d(
            x,
            weight,
            args[2],
            stride=options.get("stride", 1),
            padding=options["padding"],
            output_padding=options["output_padding"],
        )
    if name == "group_norm_ncdhw":
        assert "num_groups" in options and "eps" in options
        return F.group_norm(x, options["num_groups"], args[1], args[2], eps=options["eps"])
    if name == "avg_pool3d_ncdhw":
        return F.avg_pool3d(x, options.get("kernel_size", 2), options.get("stride", 2))
    if name in ("upsample_nearest_ncdhw", "upsample_trilinear_ncdhw"):
        mode = "nearest" if "nearest" in name else "trilinear"
        align = None if mode == "nearest" else False
        return F.interpolate(
            x, scale_factor=options.get("scale", 2), mode=mode, align_corners=align
        )
    if name == "pad_spatial_trailing_ncdhw":
        return F.pad(x, (0, 1) * 3)
    if name == "concat_channels_ncdhw":
        right = args[1]
        assert right is not None
        return torch.cat([x, right], dim=1)
    raise ValueError(f"No PyTorch reference for {name}")


def reference(
    name: str, case: ArrayCase, arrays: dict[str, HostArray]
) -> torch.Tensor | HostArray | tuple[HostArray, HostArray] | None:
    host = [None if key is None else arrays[key] for key in case.args]
    if name == "layout":
        return layout_reference(case, host)
    if name == "precision" and case.function == "eps":
        high = np.float32(1e-5)
        return np.array([high, 1e-5 - np.float64(high)], dtype=np.float32)
    if name == "precision" and case.function == "_moments":
        source = host[0]
        assert source is not None, "The moments case requires an input array"
        grouped = source.astype(np.float64).reshape(source.shape[0], -1)
        return grouped.mean(axis=1).astype(np.float32), grouped.var(axis=1).astype(np.float32)
    args = [None if value is None else from_numpy(value) for value in host]
    if name == "ops":
        return ops_reference(case, args)
    x = args[0]
    assert x is not None, "Tensor reference cases require an input tensor"
    if case.function == "deconv2x_ncdhw":
        weight = args[1]
        assert weight is not None, "The deconvolution case requires a weight tensor"
        return F.conv_transpose3d(x, weight, args[2], stride=2)
    if case.function == "upsample_add_ncdhw":
        skip = args[1]
        assert skip is not None, "The upsample-add case requires a skip tensor"
        return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False) + skip
    if name == "precision" and case.function == "conv":
        weight = args[1]
        assert weight is not None, "The convolution case requires a weight tensor"
        options = case.options
        assert "padding" in options and "stride" in options
        return F.conv3d(x, weight, args[2], padding=options["padding"], stride=options["stride"])
    if name == "precision" and case.function == "norm":
        return F.group_norm(x, 1, args[1], args[2], eps=1e-5)
    raise ValueError(f"No reference for {name}.{case.function}")


def main() -> None:
    root = output_directory(__doc__)
    arrays, cases = array_cases()
    with torch.no_grad():
        for name in ("layout", "ops", "upsample", "precision"):
            payload = dict(arrays)
            recorded: list[CaseRecord] = []
            for case in cases[name]:
                if case.error is not None:
                    continue  # Package input contracts are tested directly.
                value = reference(name, case, arrays)
                values = value if isinstance(value, tuple) else (() if value is None else (value,))
                keys: list[str] = []
                for index, result in enumerate(values):
                    key = f"{case.id}__output_{index}"
                    payload[key] = cast(HostArray, np.asarray(result))
                    keys.append(key)
                recorded.append({"id": case.id, "outputs": keys})
            save_fixture(root, name, 7081, payload, cases=recorded)


if __name__ == "__main__":
    main()
