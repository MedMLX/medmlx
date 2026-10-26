"""MAISI-v2 label vocabulary and grid contracts shared by model providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

MaisiLabelKind = Literal["anatomy", "pathology", "envelope", "internal"]

# Grid sizes (voxels) and spacing ranges (mm) the MAISI mask-to-CT pipeline accepts.
MAISI_VALID_XY = frozenset({256, 384, 512})
MAISI_VALID_Z = frozenset({128, 256, 384, 512, 640, 768})

_LABEL_NAMES: tuple[str, ...] = (
    "background",
    "liver",
    "dummy1",
    "spleen",
    "pancreas",
    "right kidney",
    "aorta",
    "inferior vena cava",
    "right adrenal gland",
    "left adrenal gland",
    "gallbladder",
    "esophagus",
    "stomach",
    "duodenum",
    "left kidney",
    "bladder",
    "dummy2",
    "portal vein and splenic vein",
    "dummy3",
    "small bowel",
    "dummy4",
    "dummy5",
    "brain",
    "lung tumor",
    "pancreatic tumor",
    "hepatic vessel",
    "hepatic tumor",
    "colon cancer primaries",
    "left lung upper lobe",
    "left lung lower lobe",
    "right lung upper lobe",
    "right lung middle lobe",
    "right lung lower lobe",
    "vertebrae L5",
    "vertebrae L4",
    "vertebrae L3",
    "vertebrae L2",
    "vertebrae L1",
    "vertebrae T12",
    "vertebrae T11",
    "vertebrae T10",
    "vertebrae T9",
    "vertebrae T8",
    "vertebrae T7",
    "vertebrae T6",
    "vertebrae T5",
    "vertebrae T4",
    "vertebrae T3",
    "vertebrae T2",
    "vertebrae T1",
    "vertebrae C7",
    "vertebrae C6",
    "vertebrae C5",
    "vertebrae C4",
    "vertebrae C3",
    "vertebrae C2",
    "vertebrae C1",
    "trachea",
    "left iliac artery",
    "right iliac artery",
    "left iliac vena",
    "right iliac vena",
    "colon",
    "left rib 1",
    "left rib 2",
    "left rib 3",
    "left rib 4",
    "left rib 5",
    "left rib 6",
    "left rib 7",
    "left rib 8",
    "left rib 9",
    "left rib 10",
    "left rib 11",
    "left rib 12",
    "right rib 1",
    "right rib 2",
    "right rib 3",
    "right rib 4",
    "right rib 5",
    "right rib 6",
    "right rib 7",
    "right rib 8",
    "right rib 9",
    "right rib 10",
    "right rib 11",
    "right rib 12",
    "left humerus",
    "right humerus",
    "left scapula",
    "right scapula",
    "left clavicula",
    "right clavicula",
    "left femur",
    "right femur",
    "left hip",
    "right hip",
    "sacrum",
    "left gluteus maximus",
    "right gluteus maximus",
    "left gluteus medius",
    "right gluteus medius",
    "left gluteus minimus",
    "right gluteus minimus",
    "left autochthon",
    "right autochthon",
    "left iliopsoas",
    "right iliopsoas",
    "left atrial appendage",
    "brachiocephalic trunk",
    "left brachiocephalic vein",
    "right brachiocephalic vein",
    "left common carotid artery",
    "right common carotid artery",
    "costal cartilages",
    "heart",
    "left kidney cyst",
    "right kidney cyst",
    "prostate",
    "pulmonary vein",
    "skull",
    "spinal cord",
    "sternum",
    "left subclavian artery",
    "right subclavian artery",
    "superior vena cava",
    "thyroid gland",
    "vertebrae S1",
    "bone lesion",
    "dummy6",
    "dummy7",
    "dummy8",
    "airway",
)

_PATHOLOGY_LABELS = frozenset({23, 24, 26, 27, 116, 117, 128})
_ALIASES = {
    "hcc": "hepatic tumor",
    "liver tumor": "hepatic tumor",
    "liver cancer": "hepatic tumor",
    "hepatic vessels": "hepatic vessel",
    "portal vein": "portal vein and splenic vein",
    "splenic vein": "portal vein and splenic vein",
    "ivc": "inferior vena cava",
    "colon tumor": "colon cancer primaries",
    "colon cancer": "colon cancer primaries",
    "left iliac vein": "left iliac vena",
    "right iliac vein": "right iliac vena",
    "left clavicle": "left clavicula",
    "right clavicle": "right clavicula",
    "body envelope": "body",
}


@dataclass(frozen=True, slots=True)
class MaisiLabel:
    """One canonical class in an rflow-CT label map."""

    value: int
    name: str
    kind: MaisiLabelKind


def _label_kind(value: int, name: str) -> MaisiLabelKind:
    if name.startswith("dummy") or value == 0:
        return "internal"
    if value == 200:
        return "envelope"
    if value in _PATHOLOGY_LABELS:
        return "pathology"
    return "anatomy"


MAISI_LABELS: tuple[MaisiLabel, ...] = tuple(
    MaisiLabel(value, name, _label_kind(value, name))
    for value, name in (*enumerate(_LABEL_NAMES), (200, "body"))
)
"""Complete canonical MAISI-v2 vocabulary, including internal labels."""

MAISI_ANATOMY_NAMES: tuple[str, ...] = tuple(
    label.name for label in MAISI_LABELS if label.kind != "internal"
)
"""Selectable MAISI anatomy, pathology, and body-envelope names."""

_LABELS_BY_VALUE = {label.value: label for label in MAISI_LABELS}
_LABELS_BY_NAME = {label.name.casefold(): label for label in MAISI_LABELS}


def list_maisi_labels(*, include_internal: bool = False) -> tuple[MaisiLabel, ...]:
    """Return the supported MAISI labels in numeric order."""

    if include_internal:
        return MAISI_LABELS
    return tuple(label for label in MAISI_LABELS if label.kind != "internal")


def maisi_label(value: int | str) -> MaisiLabel:
    """Resolve a label id, canonical name, or common alias (for example ``ivc``).

    Names are matched case-insensitively with ``_`` and ``-`` treated as spaces.
    The internal labels (``background`` and the ``dummy*`` placeholders) are
    rejected by name but resolve by id.

    Raises:
        ValueError: If the id or name is not a known MAISI label.
    """

    if isinstance(value, int):
        label = _LABELS_BY_VALUE.get(value)
        if label is None:
            raise ValueError(f"Unknown MAISI label value: {value}")
        return label
    normalized = " ".join(value.replace("_", " ").replace("-", " ").split()).casefold()
    normalized = _ALIASES.get(normalized, normalized)
    label = _LABELS_BY_NAME.get(normalized)
    if label is not None and label.kind != "internal":
        return label
    raise ValueError(f"Unknown MAISI anatomy: {value!r}")


def normalize_maisi_anatomy_name(value: str) -> str:
    """Return the upstream canonical name for one user-facing anatomy name."""

    return maisi_label(value).name


def validate_maisi_geometry(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> None:
    """Validate the grid accepted by MAISI-v2 mask-to-CT generation.

    ``shape`` is the label array shape in voxels ``(X, Y, Z)`` and ``spacing`` the
    voxel size in mm along those axes. X and Y must be equal, 256, 384 or 512, and Z
    one of 128, 256, 384, 512, 640 or 768. X/Y spacing must match within 1e-4 mm and
    lie in 0.5..3.0 mm; Z spacing in 0.5..5.0 mm.

    Raises:
        ValueError: If the shape or spacing is outside those limits.
    """

    if shape[0] != shape[1] or shape[0] not in MAISI_VALID_XY or shape[2] not in MAISI_VALID_Z:
        raise ValueError(
            "MAISI mask shape must be X=Y in {256, 384, 512} and "
            "Z in {128, 256, 384, 512, 640, 768}; "
            f"got {shape}"
        )
    if abs(spacing[0] - spacing[1]) > 1e-4:
        raise ValueError(f"MAISI mask X/Y spacing must match; got {spacing[:2]}")
    if not 0.5 <= spacing[0] <= 3.0 or not 0.5 <= spacing[2] <= 5.0:
        raise ValueError(
            f"MAISI mask spacing must be X/Y 0.5..3.0 mm and Z 0.5..5.0 mm; got {spacing}"
        )


NV_GENERATE_CT_SIZE_CONTROL_ORGANS = frozenset(
    {"liver", "gallbladder", "stomach", "pancreas", "colon"}
)


NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES = frozenset(
    {
        "hepatic tumor",
        "bone lesion",
        "lung tumor",
        "colon cancer primaries",
        "pancreatic tumor",
    }
)


NV_GENERATE_CT_SIZE_CONTROL_TARGETS = (
    NV_GENERATE_CT_SIZE_CONTROL_ORGANS | NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES
)


def _require_finite(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class NvGenerateAnatomySize:
    """One supported NV-Generate-CT size target and its conditioning scale.

    ``anatomy`` is normalized to the canonical MAISI name. ``scale`` is the value
    placed in that anatomy's slot of the 10-element size-condition vector: 0..1,
    or -1, which marks a slot as absent (a tumor slot equal to -1 is not targeted).
    """

    anatomy: str
    scale: float

    def __post_init__(self) -> None:
        if not self.anatomy.strip():
            raise ValueError("NV-Generate-CT anatomy name must not be empty")
        anatomy = normalize_maisi_anatomy_name(self.anatomy)
        if anatomy not in NV_GENERATE_CT_SIZE_CONTROL_TARGETS:
            choices = ", ".join(sorted(NV_GENERATE_CT_SIZE_CONTROL_TARGETS))
            raise ValueError(f"NV-Generate-CT size control supports only: {choices}")
        object.__setattr__(self, "anatomy", anatomy)
        _require_finite("NV-Generate-CT anatomy scale", self.scale)
        if self.scale != -1.0 and not 0.0 <= self.scale <= 1.0:
            raise ValueError("NV-Generate-CT anatomy scale must be -1 or between 0 and 1")


__all__ = [
    "MAISI_ANATOMY_NAMES",
    "MAISI_LABELS",
    "MAISI_VALID_XY",
    "MAISI_VALID_Z",
    "NV_GENERATE_CT_SIZE_CONTROL_ORGANS",
    "NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES",
    "NV_GENERATE_CT_SIZE_CONTROL_TARGETS",
    "MaisiLabel",
    "MaisiLabelKind",
    "NvGenerateAnatomySize",
    "list_maisi_labels",
    "maisi_label",
    "normalize_maisi_anatomy_name",
    "validate_maisi_geometry",
]
