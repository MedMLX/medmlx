"""Build and validate the file descriptions returned by model runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict, cast

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


class OutputEntry(TypedDict):
    """One output file and its runtime-declared meaning."""

    path: str
    kind: str
    coordinate_space: str
    labels: NotRequired[list[OutputLabel] | None]
    grid_of: NotRequired[str | None]
    units: NotRequired[str | None]


def output_entry(
    path: str | Path, *, kind: str, coordinate_space: str,
    labels: list[OutputLabel] | None = None, grid_of: str | None = None,
    units: str | None = None,
) -> OutputEntry:
    """Build one JSON output entry, raising ValueError for invalid metadata.

    Paths retain their relative or absolute form. No files are read. Mask kinds
    require a label list; an empty list describes a run with no named labels.
    ``grid_of`` names an input argument, and ``units`` states the value units.
    """
    entry: OutputEntry = {
        "path": path.as_posix() if isinstance(path, Path) else path,
        "kind": kind,
        "coordinate_space": coordinate_space,
        "labels": labels,
        "grid_of": grid_of,
        "units": units,
    }
    _validate_entry(entry, "output")
    return entry


def validate_outputs(value: object) -> list[OutputEntry]:
    """Return a checked outputs list, or raise ValueError with the failing field.

    The list and its entries are left unchanged. Unknown fields are ignored.
    """
    if not isinstance(value, list):
        raise ValueError("outputs must be a list")
    entries = cast(list[object], value)
    for index, entry in enumerate(entries):
        _validate_entry(entry, f"outputs[{index}]")
    return cast(list[OutputEntry], value)


def _validate_entry(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    entry = cast(dict[str, object], value)
    _text(entry.get("path"), f"{field}.path")
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


__all__ = ["OutputEntry", "OutputLabel", "output_entry", "validate_outputs"]
