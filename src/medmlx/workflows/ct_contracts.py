"""Finite, closed schemas for CT tool requests and their spatial evidence."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Contract(BaseModel):
    """Reject coercion, nonfinite numbers and undeclared model-generated arguments."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Detect(Contract):
    """Ask the host to run one lung-nodule detection pass over the working CT."""

    action: Literal["detect_nodules"]


class Segment(Contract):
    """Ask the host to segment and measure one candidate returned by detection."""

    action: Literal["segment_candidate"]
    candidate_id: Annotated[str, Field(pattern=r"^candidate-[0-9]{3,}$")]


class Finish(Contract):
    """End tool use; interpretation is generated separately after this action."""

    action: Literal["finish"]


type Action = Detect | Segment | Finish


class Limits(Contract):
    """Caller-owned limits; the language model cannot change execution policy."""

    prompt: Annotated[str, Field(min_length=1, max_length=8192)]
    score_threshold: Annotated[FiniteFloat, Field(ge=0, le=1)]
    max_candidates: Annotated[int, Field(ge=1, le=20)]
    max_segments: Annotated[int, Field(ge=1, le=20)]
    max_steps: Annotated[int, Field(ge=3, le=32)]
    max_new_tokens: Annotated[int, Field(ge=64, le=8192)]
    padding_mm: Annotated[FiniteFloat, Field(ge=0, le=30)]
    assume_mm: bool


class Detection(Contract):
    """One detector row: a RAS centre/size box in millimetres and its class score."""

    box: Annotated[list[FiniteFloat], Field(min_length=6, max_length=6)]
    label: Literal[0]
    score: Annotated[FiniteFloat, Field(ge=0, le=1)]
    source_image: str


class Detections(Contract):
    """The lung detector's published RAS centre/size record, without reinterpretation."""

    schema_version: Literal["medmlx.detections.v1"]
    model_id: Literal["lung_nodule_ct_detection"]
    source_format: str
    coordinate_space: Literal["image_physical"]
    coordinate_convention: Literal["nifti_ras"]
    source_coordinate_space: Literal["dicom_patient_lps"]
    units: Literal["mm"]
    box_mode: Literal["cccwhd"]
    classes: dict[str, str]
    source_output: str
    detections: list[Detection]


class Candidate(Contract):
    """A detector row grounded on the saved orthogonal RAS working CT."""

    candidate_id: str
    detection_index: int
    score: float
    box_ras_mm: list[float]
    box_xyxy: list[float]
    frame_idx: int
    slice_range: list[int]
    clipped_to_image: bool


class Measurement(Contract):
    """Mask-derived quantities on the original grid; diameter is volume-equivalent."""

    candidate_id: str
    voxel_count: int
    volume_mm3: float
    equivalent_sphere_diameter_mm: float
    centroid_ras_mm: list[float]
    mask_path: str
