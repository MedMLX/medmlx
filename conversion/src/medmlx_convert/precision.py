"""Explicit mixed FP16 authoring with FP32 public tensors and normalization.

This is a numerical build strategy, not a claim about accelerator placement.
Functional normalization/softmax in upstream models is unchanged; unsupported
dtype combinations fail during eager replay rather than falling back to FP32.
"""

import copy
import os
from collections.abc import Iterator
from enum import StrEnum
from typing import cast

import torch

from medmlx_convert.boundary import (
    Module,
    as_sequence,
    as_tensor,
    as_tensors,
    attribute,
    function,
    library,
    module_class,
)


class SourceDevice(StrEnum):
    """Device that runs the upstream PyTorch source for graph capture and replay."""

    CPU = "cpu"
    MPS = "mps"


class Float32Norm(Module):
    """Compute a copied normalization in FP32 and restore the incoming dtype."""

    def __init__(self, norm: torch.nn.Module) -> None:
        super().__init__()
        self.norm = copy.deepcopy(norm).float().eval()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return as_tensor(self.norm(value.float())).to(value.dtype)


class StableLogSoftmax(Module):
    """Avoid FP16 log(softmax) underflow by evaluating x-logsumexp(x) in FP32."""

    def __init__(self, source: torch.nn.LogSoftmax) -> None:
        super().__init__()
        if source.dim is None:
            raise ValueError("FP16 authoring requires an explicit LogSoftmax axis")
        self.dim = source.dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return value - torch.logsumexp(value, dim=self.dim, keepdim=True)


class Float16Network(Module):
    """Cast floating public inputs to FP16 and floating outputs back to FP32."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = tuple(value.half() if value.is_floating_point() else value for value in values)
        result: object = self.model(*inputs)
        outputs = (result,) if isinstance(result, torch.Tensor) else as_tensors(result)
        return tuple(value.float() if value.is_floating_point() else value for value in outputs)


def _configure_keras_fp16(model: torch.nn.Module) -> None:
    """Align copied Keras layers' public compute policies with their FP16 tensors.

    Torch .half() changes storage but leaves Keras dtype policies unchanged.
    Keras would otherwise cast inputs back to FP32 before FP16 convolution.
    Keras normalization retains its upstream implementation, including its
    internal FP32 arithmetic; its rounded weights remain part of precision loss.
    Ordinary Torch recipes do not import Keras.
    """
    if not any(base.__module__ == "keras.src.layers.layer" and base.__name__ == "Layer"
               for child in model.modules() for base in type(child).__mro__):
        return
    layer_type = module_class(attribute(library("keras"), "layers"), "Layer")
    for child in model.modules():
        if isinstance(child, layer_type):
            # Keras Torch tracks variables in a ParameterDict; Module.modules()
            # does not expose graph layers. Use the installed Keras traversal.
            layers = as_sequence(function(child, "_flatten_layers")(
                include_self=True, recursive=True,
            ))
            for layer in layers:
                if not isinstance(layer, layer_type):
                    raise TypeError("Keras traversal returned a non-Layer object")
                function(layer, "__setattr__")("dtype_policy", "float16")


def prepare_fp16(
    model: torch.nn.Module, device: SourceDevice,
    fp32_norm_types: tuple[type[torch.nn.Module], ...] = (),
) -> torch.nn.Module:
    """Copy the model, retain original FP32 norm bytes, and author mixed precision.

    Exact standard normalization types and recipe-declared custom types are
    wrapped. Other custom types keep their implementation. Require the selected
    source device with no CPU fallback.
    The original FP32 oracle is never mutated.
    """
    if device == SourceDevice.MPS and not torch.backends.mps.is_available():
        raise RuntimeError("FP16 source requires MPS; select --source-device cpu explicitly")
    if device == SourceDevice.MPS and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK; implicit CPU fallback is prohibited")
    norms = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
             torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
             torch.nn.GroupNorm, torch.nn.LayerNorm, *fp32_norm_types)
    replacements: dict[str, torch.nn.Module] = {}
    modules = cast(Iterator[tuple[str, torch.nn.Module]],
                   function(model, "named_modules")(remove_duplicate=False))
    for name, module in modules:
        if type(module) in norms:
            replacements[name] = Float32Norm(module)
        elif type(module) is torch.nn.LogSoftmax:
            replacements[name] = StableLogSoftmax(module)
    prepared = copy.deepcopy(model).half().eval()
    _configure_keras_fp16(prepared)
    for name, replacement in replacements.items():
        if name:
            prepared.set_submodule(name, replacement, strict=True)
        else:
            prepared = replacement
    return Float16Network(prepared).eval().to(device.value)
