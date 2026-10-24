"""Optional Torch checkpoint deserialization; install the conversion extra."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError, MissingDependencyError

type RawCheckpoint = dict[object, object]


def load_torch_checkpoint(path: Path, *, weights_only: bool = True) -> RawCheckpoint:
    """Deserialize a source checkpoint on the host with ``torch.load``.

    No model is built or executed. ``weights_only=True`` (the default) restricts
    unpickling to tensors and plain containers; pass ``False`` only for a
    checkpoint from a trusted source, since that setting can run arbitrary code.

    Args:
        path: Checkpoint file to read.
        weights_only: Forwarded to ``torch.load``.

    Returns:
        The top-level mapping, loaded onto the CPU. Its values are not validated.

    Raises:
        MissingDependencyError: Torch is not installed.
        InvalidInputError: The file does not deserialize to a dict.
    """

    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "Checkpoint conversion requires torch to unpickle the source files",
            extra="conversion",
            hint="Install medmlx-core[conversion] and rerun the converter.",
        ) from exc
    payload: object = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(payload, dict):
        raise InvalidInputError(f"{path.name} is not a mapping checkpoint")
    # Deserialization proves only a raw mapping; tensor_mapping_from_payload parses it.
    return cast(RawCheckpoint, payload)


__all__ = ["load_torch_checkpoint"]
