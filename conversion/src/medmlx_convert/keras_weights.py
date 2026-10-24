"""Strict legacy Keras HDF5 to native Torch parameter conversion.

Source order is the explicit Keras layer_names/weight_names topology, never a
lexicographic HDF5 traversal. The caller supplies the matching native architecture.
Every declared model weight and target parameter must correspond; nothing is skipped.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from medmlx_convert.boundary import as_tensor, function


class Layout(StrEnum):
    VECTOR = "vector"
    DENSE = "dense_io_to_oi"
    CONV3D = "conv3d_dhwio_to_oidhw"


@dataclass(frozen=True)
class WeightBinding:
    """A source dataset's correspondence to one native parameter and its layout change."""

    source: str
    target: str
    layout: Layout


def _names(value: object) -> tuple[str, ...]:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError("Keras layer and weight names must be one-dimensional arrays")
    names: list[str] = []
    for entry in cast(npt.NDArray[np.str_ | np.bytes_], value):
        if isinstance(entry, bytes):
            names.append(entry.decode("utf-8"))
        elif isinstance(entry, str):
            names.append(entry)
        else:
            raise ValueError("Keras weight names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate Keras layer or weight names")
    return tuple(names)


def _datasets(source: h5py.File) -> tuple[str, ...]:
    root = source["model_weights"] if "model_weights" in source else source
    if not isinstance(root, h5py.Group):
        raise ValueError("Keras model_weights must be a group")
    datasets: list[str] = []
    for layer_name in _names(root.attrs["layer_names"]):
        layer = root[layer_name]
        if not isinstance(layer, h5py.Group):
            raise ValueError(f"Keras layer {layer_name} must be a group")
        for weight_name in _names(layer.attrs["weight_names"]):
            weight = layer[weight_name]
            if not isinstance(weight, h5py.Dataset):
                raise ValueError(f"Keras weight {weight_name} must be a dataset")
            datasets.append(weight.name)
    if len(set(datasets)) != len(datasets):
        raise ValueError("Keras topology lists the same dataset more than once")
    return tuple(datasets)


def _targets(module: nn.Module) -> tuple[tuple[str, Layout], ...]:
    targets: list[tuple[str, Layout]] = []
    # Torch 2.8 leaves part of named_modules' recursive iterator unannotated.
    modules = cast(Iterator[tuple[str, nn.Module]], function(module, "named_modules")())
    for name, layer in modules:
        parameters = tuple(layer.named_parameters(recurse=False))
        if not parameters:
            continue
        if isinstance(layer, nn.Conv3d):
            layout = Layout.CONV3D
        elif isinstance(layer, nn.Linear):
            layout = Layout.DENSE
        else:
            raise ValueError(f"Unsupported trainable layer {name}: {type(layer).__name__}")
        prefix = f"{name}." if name else ""
        for parameter, _ in parameters:
            if parameter not in {"weight", "bias"}:
                raise ValueError(f"Unsupported parameter {prefix}{parameter}")
            targets.append((prefix + parameter, layout if parameter == "weight" else Layout.VECTOR))
    return tuple(targets)


def load_keras_hdf5_weights(module: nn.Module, path: Path) -> tuple[WeightBinding, ...]:
    """Load a complete matching Conv3D/Dense Keras checkpoint, returning all correspondences.

    This conversion is CPU FP32 only and copies values without precision changes.
    The destination is mutated only after every tensor has passed admission, so an
    incompatible or corrupt checkpoint cannot leave a partially initialized model.
    """
    targets = _targets(module)
    expected = module.state_dict()
    if any(value.dtype != torch.float32 or value.device.type != "cpu"
           for value in expected.values()):
        raise ValueError("Keras conversion requires a CPU FP32 target model")
    converted: dict[str, Tensor] = {}
    bindings: list[WeightBinding] = []
    with h5py.File(str(path), "r") as source:
        datasets = _datasets(source)
        if len(datasets) != len(targets) or len(expected) != len(targets):
            raise ValueError(f"Checkpoint weights: {len(datasets)}; required: {len(targets)}")
        for dataset, (target, layout) in zip(datasets, targets, strict=True):
            node = source[dataset]
            assert isinstance(node, h5py.Dataset)
            if node.dtype != np.dtype("float32"):
                raise ValueError(f"Expected FP32 source weight at {dataset}, got {node.dtype}")
            array = cast(npt.NDArray[np.float32], node[()])
            if layout is Layout.CONV3D:
                array = array.transpose(4, 3, 0, 1, 2)
            elif layout is Layout.DENSE:
                array = array.T
            tensor = as_tensor(function(torch, "from_numpy")(np.ascontiguousarray(array)))
            if tensor.shape != expected[target].shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid shape or nonfinite weight: {dataset} -> {target}")
            converted[target] = tensor
            bindings.append(WeightBinding(dataset, target, layout))
    module.load_state_dict(converted, strict=True)
    return tuple(bindings)
