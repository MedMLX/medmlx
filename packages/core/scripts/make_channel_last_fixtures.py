#!/usr/bin/env python3
"""Record channel-last primitives and residual blocks from pinned MONAI/PyTorch."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from monai.networks.blocks.activation import Swish
from monai.networks.blocks.dynunet_block import UnetResBlock
from reference_support import HostArray, output_directory, save_fixture
from torch.nn import functional as F

from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])


def seeded_arrays() -> tuple[np.random.Generator, dict[str, HostArray]]:
    rng = np.random.default_rng(20261007)
    arrays: dict[str, HostArray] = {
        "input_2d": rng.normal(size=(1, 5, 7, 4)).astype(np.float32),
        "conv.weight": rng.normal(size=(6, 2, 3, 3)).astype(np.float32),
        "conv.bias": rng.normal(size=(6,)).astype(np.float32),
        "bn.weight": rng.normal(size=(6,)).astype(np.float32),
        "bn.bias": rng.normal(size=(6,)).astype(np.float32),
        "bn.running_mean": rng.normal(size=(6,)).astype(np.float32),
        "bn.running_var": rng.uniform(0.1, 1, size=(6,)).astype(np.float32),
        "input_3d": rng.normal(size=(1, 3, 4, 5, 2)).astype(np.float32),
        "deconv.weight": rng.normal(size=(2, 3, 2, 2, 2)).astype(np.float32),
    }
    return rng, arrays


def record_layers(arrays: dict[str, HostArray]) -> None:
    tensors = {key: from_numpy(value) for key, value in arrays.items()}
    x = tensors["input_2d"].permute(0, 3, 1, 2)
    x = F.conv2d(x, tensors["conv.weight"], tensors["conv.bias"], padding=1, groups=2)
    arrays["expected_conv"] = x.permute(0, 2, 3, 1).numpy()
    x = F.batch_norm(
        x,
        tensors["bn.running_mean"],
        tensors["bn.running_var"],
        tensors["bn.weight"],
        tensors["bn.bias"],
        training=False,
        eps=1e-3,
    )
    arrays["expected_bn"] = x.permute(0, 2, 3, 1).numpy()
    x = Swish()(x).permute(0, 2, 3, 1)
    arrays["expected_swish"] = x.numpy()
    x = F.layer_norm(x, (6,), eps=1e-5)
    arrays["expected_ln"] = x.numpy()
    arrays["expected_2d"] = x.numpy()
    z = F.conv_transpose3d(
        tensors["input_3d"].permute(0, 4, 1, 2, 3), tensors["deconv.weight"], stride=2
    )
    arrays["expected_deconv"] = z.permute(0, 2, 3, 4, 1).numpy()
    arrays["expected_3d"] = F.instance_norm(z, eps=1e-5).permute(0, 2, 3, 4, 1).numpy()
    arrays["expected_relu"] = F.relu(tensors["input_2d"]).numpy()


def record_extra_layers(rng: np.random.Generator, arrays: dict[str, HostArray]) -> None:
    for name, shape in (
        ("projection.weight", (3, 4)),
        ("projection.bias", (3,)),
        ("ln.weight", (4,)),
        ("ln.bias", (4,)),
        ("volume.weight", (3, 2, 3, 3, 3)),
    ):
        arrays[name] = rng.normal(size=shape).astype(np.float32)
    tensors = {key: from_numpy(value) for key, value in arrays.items()}
    arrays["expected_linear"] = F.linear(
        tensors["input_2d"], tensors["projection.weight"], tensors["projection.bias"]
    ).numpy()
    arrays["expected_affine_ln"] = F.layer_norm(
        tensors["input_2d"], (4,), tensors["ln.weight"], tensors["ln.bias"], eps=1e-5
    ).numpy()
    y = F.conv3d(tensors["input_3d"].permute(0, 4, 1, 2, 3), tensors["volume.weight"], padding=1)
    arrays["expected_volume"] = y.permute(0, 2, 3, 4, 1).numpy()


def record_residuals(rng: np.random.Generator, arrays: dict[str, HostArray]) -> None:
    x = from_numpy(arrays["input_3d"]).permute(0, 4, 1, 2, 3)
    for name, channels in (("residual_same", 2), ("residual_project", 3)):
        block = UnetResBlock(3, 2, channels, 3, 1, ("instance", {"affine": False, "eps": 1e-5}))
        state: dict[str, torch.Tensor] = {}
        for key, value in cast(dict[str, torch.Tensor], block.state_dict()).items():
            array = rng.normal(size=tuple(value.shape)).astype(np.float32)
            arrays[f"{name}.{key}"] = array
            state[key] = from_numpy(array)
        block.load_state_dict(state, strict=True)
        arrays[f"expected_{name}"] = block(x).permute(0, 2, 3, 4, 1).numpy()


def main() -> None:
    root = output_directory(__doc__)
    rng, arrays = seeded_arrays()
    with torch.no_grad():
        record_layers(arrays)
        record_extra_layers(rng, arrays)
        record_residuals(rng, arrays)
    save_fixture(root, "channel_last", 20261007, arrays)


if __name__ == "__main__":
    main()
