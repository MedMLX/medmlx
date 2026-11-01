"""Typed boundary for MLX calls whose installed Python stubs are incomplete."""

import mlx.core as mx


def invoke(owner: object, name: str, *args: object, **kwargs: object) -> object:
    value = getattr(owner, name)
    if not callable(value):
        raise TypeError(f"MLX {name} is not callable")
    return value(*args, **kwargs)


def tensor(value: object) -> mx.array:
    if not isinstance(value, mx.array):
        raise TypeError("MLX operation did not return a tensor")
    return value
