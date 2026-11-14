"""Check the files a model run reports and the metadata that describes them.

A runner result names its files in ``outputs``, a map from role to path, so
callers read ``result["outputs"]["mask"]``. Optional ``artifacts`` describe
the same roles: what each file holds, its coordinate space, label names, grid
and units. Every artifact role must also appear in ``outputs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, NotRequired, TypedDict, cast

# RadNN's OutputKind and CoordinateSpace vocabulary.
_OUTPUT_KINDS = frozenset({
    "binary_mask", "label_map", "point_set", "centerline", "detection_set",
    "transform", "correspondence_set", "feature_tensor", "measurement", "report",
    "preview", "registered_image", "synthetic_image", "class_label", "scalar_field",
})
_COORDINATE_SPACES = frozenset({
    "image_voxel", "image_physical", "patient", "tracker", "instrument", "pixel",
    "unresolved", "not_applicable",
})


class OutputLabel(TypedDict):
    """One non-negative label value and its name for this run."""

    value: int
    name: str


class Artifact(TypedDict):
    """What one output file holds, keyed by the same role as its path."""

    kind: str
    coordinate_space: str
    labels: NotRequired[list[OutputLabel] | None]
    grid_of: NotRequired[str | None]
    units: NotRequired[str | None]


class CheckedOutputs(NamedTuple):
    """A result's role-to-path map and its artifact metadata by role."""

    paths: dict[str, str]
    artifacts: dict[str, Artifact]


def artifact(
    *, kind: str, coordinate_space: str, labels: list[OutputLabel] | None = None,
    grid_of: str | None = None, units: str | None = None,
) -> Artifact:
    """Build one artifact description, raising ValueError for invalid metadata.

    Mask kinds require a label list; an empty list describes a run with no
    named labels. ``grid_of`` names the input argument whose grid the file
    shares, and ``units`` states its value units. No files are read.
    """
    entry: Artifact = {
        "kind": kind,
        "coordinate_space": coordinate_space,
        "labels": labels,
        "grid_of": grid_of,
        "units": units,
    }
    _validate_artifact(entry, "artifact")
    return entry


def validate_outputs(result: Mapping[str, object]) -> CheckedOutputs:
    """Return a result's checked ``outputs`` and ``artifacts``, or raise ValueError.

    A missing field reads as an empty map. Paths stay as given, relative or
    absolute, and no files are read. The returned maps are the result's own
    objects, not copies. Unknown fields inside an artifact are ignored.
    """
    paths = _mapping(result.get("outputs", {}), "outputs")
    for role, path in paths.items():
        _text(role, "outputs role")
        _text(path, f"outputs.{role}")
    artifacts = _mapping(result.get("artifacts", {}), "artifacts")
    for role, entry in artifacts.items():
        if role not in paths:
            raise ValueError(f"artifacts.{role} has no matching outputs.{role}")
        _validate_artifact(entry, f"artifacts.{role}")
    return CheckedOutputs(cast(dict[str, str], paths), cast(dict[str, Artifact], artifacts))


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object keyed by output role")
    return cast(dict[str, object], value)


def _validate_artifact(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    entry = cast(dict[str, object], value)
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind not in _OUTPUT_KINDS:
        raise ValueError(f"{field}.kind must be one of: {', '.join(sorted(_OUTPUT_KINDS))}")
    space = entry.get("coordinate_space")
    if not isinstance(space, str) or space not in _COORDINATE_SPACES:
        raise ValueError(
            f"{field}.coordinate_space must be one of: {', '.join(sorted(_COORDINATE_SPACES))}"
        )
    for name in ("grid_of", "units"):
        if entry.get(name) is not None:
            _text(entry[name], f"{field}.{name}")
    labels = entry.get("labels")
    if labels is None:
        if kind in {"label_map", "binary_mask"}:
            raise ValueError(f"{field}.labels is required for {kind}")
        return
    if not isinstance(labels, list):
        raise ValueError(f"{field}.labels must be a list")
    seen: set[int] = set()
    for index, item in enumerate(cast(list[object], labels)):
        label_field = f"{field}.labels[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label_field} must be an object")
        label = cast(dict[str, object], item)
        number = label.get("value")
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{label_field}.value must be a non-negative integer")
        if number in seen:
            raise ValueError(f"{label_field}.value duplicates label value {number}")
        seen.add(number)
        _text(label.get("name"), f"{label_field}.name")


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


__all__ = ["Artifact", "CheckedOutputs", "OutputLabel", "artifact", "validate_outputs"]
