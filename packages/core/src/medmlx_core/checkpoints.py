"""Host checkpoint deserialization shared by native model converters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from medmlx_core.conversion import load_torch_checkpoint
from medmlx_core.errors import InvalidInputError

HostArray = NDArray[np.generic]


def mapping_from_pairs(
    pairs: Sequence[tuple[str, HostArray]], *, what: str
) -> dict[str, HostArray]:
    """Build a tensor mapping or reject duplicate names."""

    tensors: dict[str, HostArray] = {}
    for name, array in pairs:
        if name in tensors:
            raise InvalidInputError(f"duplicate {what} parameter: {name}")
        tensors[name] = array
    return tensors


def tensor_mapping_from_payload(payload: object, *, what: str) -> dict[str, HostArray]:
    """Extract the tensor entries of a loaded checkpoint as host NumPy arrays.

    Entries that are not tensors or arrays (strings, numbers, nested containers
    and other metadata) are skipped without error, so the result may hold fewer
    names than the payload has keys. Keys are converted with ``str``.

    Args:
        payload: Mapping from ``load_torch_checkpoint`` or an equivalent object.
        what: Checkpoint label used in error messages.

    Returns:
        Parameter name to array, in payload order.

    Raises:
        InvalidInputError: ``payload`` is not a mapping, two keys map to the same
            ``str`` name, or a tensor's ``numpy()`` does not return an ndarray.
    """
    if not isinstance(payload, Mapping):
        raise InvalidInputError(f"{what} checkpoint payload must be a mapping")
    pairs: list[tuple[str, HostArray]] = []
    mapping = cast(Mapping[object, object], payload)
    for raw_name, value in mapping.items():
        name = str(raw_name)
        array = _maybe_array(value)
        if array is None:
            continue
        pairs.append((name, array))
    return mapping_from_pairs(pairs, what=what)


def _maybe_array(value: object) -> HostArray | None:
    if isinstance(value, np.ndarray):
        return cast(HostArray, value)
    if isinstance(value, (np.generic, int, float, bool, str)):
        return None
    detach = getattr(value, "detach", None)
    tensor = detach() if callable(detach) else value
    cpu = getattr(tensor, "cpu", None)
    host = cpu() if callable(cpu) else tensor
    numpy_fn = getattr(host, "numpy", None)
    if not callable(numpy_fn):
        return None
    raw_array: object = numpy_fn()
    if not isinstance(raw_array, np.ndarray):
        raise InvalidInputError("tensor numpy() must return an ndarray")
    return cast(HostArray, raw_array)


__all__ = [
    "HostArray",
    "_maybe_array",
    "load_torch_checkpoint",
    "mapping_from_pairs",
    "tensor_mapping_from_payload",
]
