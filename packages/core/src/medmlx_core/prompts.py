"""Prompt metadata shared by CPU/CUDA and MLX providers."""

from typing import Protocol


class _MedSAM2Prompt(Protocol):
    """Read-only prompt fields; validation and mode names belong to the provider."""

    @property
    def frame_idx(self) -> int: ...

    @property
    def obj_id(self) -> int: ...

    @property
    def mode(self) -> str: ...

    @property
    def box_xyxy(self) -> tuple[float, float, float, float] | None: ...

    @property
    def points_xy(self) -> tuple[tuple[float, float], ...]: ...

    @property
    def point_labels(self) -> tuple[int, ...]: ...

    @property
    def mask(self) -> object | None: ...

    @property
    def clear_old_points(self) -> bool: ...

    @property
    def normalize_coords(self) -> bool: ...


def medsam2_prompt_payload(prompt: _MedSAM2Prompt) -> dict[str, object]:
    """Serialize prompt metadata, preserving provider mode names and omitting mask voxels."""

    return {
        "frame_idx": prompt.frame_idx,
        "obj_id": prompt.obj_id,
        "prompt_mode": prompt.mode,
        "box_xyxy": [float(value) for value in prompt.box_xyxy]
        if prompt.box_xyxy is not None
        else None,
        "points_xy": [[float(x), float(y)] for x, y in prompt.points_xy],
        "point_labels": [int(value) for value in prompt.point_labels],
        "mask_supplied": prompt.mask is not None,
        "clear_old_points": prompt.clear_old_points,
        "normalize_coords": prompt.normalize_coords,
    }
