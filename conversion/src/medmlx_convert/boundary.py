"""Typed access to untyped upstream code. Every helper checks what it returns."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

Array = npt.NDArray[np.generic]


def library(name: str) -> ModuleType:
    """Import a package that ships no type information."""
    return import_module(name)


def attribute(owner: object, name: str) -> object:
    return getattr(owner, name)


def function(owner: object, name: str) -> Callable[..., object]:
    value = getattr(owner, name)
    if not callable(value):
        raise TypeError(f"{name} is {type(value).__name__}, not callable")
    return value


def class_(owner: object, name: str) -> type:
    value = getattr(owner, name)
    if not isinstance(value, type):
        raise TypeError(f"{name} is {type(value).__name__}, not a class")
    return value


def module(owner: object, name: str) -> torch.nn.Module:
    return as_module(getattr(owner, name), name)


def as_module(value: object, what: str = "value") -> torch.nn.Module:
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"{what} is {type(value).__name__}, not a torch.nn.Module")
    return value


def tensor(owner: object, name: str) -> torch.Tensor:
    return as_tensor(getattr(owner, name), name)


def as_tensor(value: object, what: str = "value") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{what} is {type(value).__name__}, not a tensor")
    return value


def as_tensors(value: object, what: str = "value") -> tuple[torch.Tensor, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{what} is {type(value).__name__}, not a sequence of tensors")
    items = cast(tuple[object, ...] | list[object], value)
    return tuple(as_tensor(item, f"{what}[{i}]") for i, item in enumerate(items))


def as_array(value: object, what: str = "value") -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{what} is {type(value).__name__}, not a NumPy array")
    return cast(Array, value)


def item(mapping: object, key: str) -> object:
    """``mapping[key]`` for an untyped mapping."""
    return function(mapping, "__getitem__")(key)


def as_int_pair(value: object, what: str = "value") -> tuple[int, int]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{what} is {type(value).__name__}, not a pair")
    items = cast(tuple[object, ...] | list[object], value)
    if len(items) != 2 or not all(isinstance(v, int) for v in items):
        raise TypeError(f"{what} is not two integers: {items}")
    return cast(int, items[0]), cast(int, items[1])


class Module(torch.nn.Module):
    """torch.nn.Module with a typed constructor; torch leaves ``Module.__init__`` unannotated."""

    def __init__(self) -> None:
        function(torch.nn.Module, "__init__")(self)


def interpolate_bilinear(
    x: torch.Tensor, scale_factor: tuple[float, float], align_corners: bool | None,
    recompute_scale_factor: bool | None,
) -> torch.Tensor:
    """Apply bilinear interpolation with the source's scale-factor semantics."""
    result = function(torch.nn.functional, "interpolate")(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    return as_tensor(result, "interpolate result")


def as_state_dict(value: object, what: str = "checkpoint") -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"{what} is {type(value).__name__}, not a state dict")
    entries = cast(dict[object, object], value)
    state: dict[str, torch.Tensor] = {}
    for key, tensor_value in entries.items():
        if not isinstance(key, str):
            raise TypeError(f"{what} has a non-string key {key!r}")
        state[key] = as_tensor(tensor_value, f"{what}[{key!r}]")
    return state


def unwrap(value: object, key: str) -> object:
    """``value[key]`` when ``value`` is a dict holding ``key``, else ``value`` itself."""
    result = value
    if isinstance(value, dict):
        entries = cast(dict[object, object], value)
        if key in entries:
            result = entries[key]
    return result


def module_class(owner: object, name: str) -> type[torch.nn.Module]:
    value = class_(owner, name)
    if not issubclass(value, torch.nn.Module):
        raise TypeError(f"{name} is not a torch.nn.Module subclass")
    return value


def as_ints(value: object, what: str = "value") -> tuple[int, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{what} is {type(value).__name__}, not a sequence of integers")
    items = cast(tuple[object, ...] | list[object], value)
    if not all(isinstance(v, int) for v in items):
        raise TypeError(f"{what} is not all integers: {items}")
    return tuple(cast(int, v) for v in items)


def as_str_dict(value: object, what: str = "value") -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{what} is {type(value).__name__}, not a mapping")
    entries = cast(dict[object, object], value)
    if not all(isinstance(k, str) for k in entries):
        raise TypeError(f"{what} has non-string keys")
    return cast(dict[str, object], entries)


def hf_download(repo: str, filename: str, revision: str) -> Path:
    """One file from a pinned Hugging Face revision."""
    path = function(library("huggingface_hub"), "hf_hub_download")(
        repo, filename, revision=revision
    )
    return Path(str(path))


def hf_snapshot(repo: str, revision: str, patterns: list[str]) -> Path:
    """The files matching ``patterns`` from a pinned Hugging Face revision."""
    path = function(library("huggingface_hub"), "snapshot_download")(
        repo, revision=revision, allow_patterns=patterns
    )
    return Path(str(path))


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return as_state_dict(function(library("safetensors.torch"), "load_file")(path), str(path))


def as_sequence(value: object, what: str = "value") -> tuple[object, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{what} is {type(value).__name__}, not a sequence")
    return tuple(cast(tuple[object, ...] | list[object], value))


def as_float(value: object, what: str = "value") -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{what} is {type(value).__name__}, not a number")
    return float(value)


def as_int(value: object, what: str = "value") -> int:
    if not isinstance(value, int):
        raise TypeError(f"{what} is {type(value).__name__}, not an integer")
    return value
