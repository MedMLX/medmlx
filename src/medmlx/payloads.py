"""Wire shapes produced by the server and its supervised workers."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JobState = Literal["running", "completed", "failed", "cancelled"]
type OutputNamesFrom = Literal["outputs", "declaration"] | None


class WorkerError(TypedDict):
    """A run's exception class name and message, as reported by the worker."""

    type: str
    message: str


class RunMemory(TypedDict):
    """Separate allocation and residency measurements, all in bytes.

    RSS sampling covers runner import through return, including result shape/key
    checks. The kernel's worker RSS peak covers its entire process lifetime.
    MLX counters retain their native scope (since startup or the runner's last
    reset); active/cache are final readings. Descendant RSS is a sum per sample,
    so shared pages can appear in multiple processes. These counters overlap:
    none is added to another or treated as macOS physical footprint. Sampling
    can miss short-lived processes or peaks, including while the GIL is held.
    ``sampling_errors`` counts inaccessible/disappearing process readings.
    """

    scope: Literal["runner_import_through_return"]
    worker_peak_rss_bytes: int
    sample_interval_seconds: float
    sampling_duration_seconds: float
    sample_count: int
    sampling_errors: int
    sampled_worker_rss_peak_bytes: int | None
    sampled_descendants_rss_sum_peak_bytes: int | None
    sampled_process_count_peak: int
    mlx_active_bytes: NotRequired[int]
    mlx_peak_bytes: NotRequired[int]
    mlx_cache_bytes: NotRequired[int]


class WorkerResponse(TypedDict):
    """The packet a worker writes: ``result`` when ``ok``, otherwise ``error``."""

    ok: bool
    result: NotRequired[JsonValue]
    error: NotRequired[WorkerError]
    measured_peak_memory_gb: NotRequired[float]
    memory_metrics: NotRequired[RunMemory]


class JobPayload(TypedDict):
    """A job as returned to clients; times are UTC ISO 8601 strings."""

    job_id: str
    model_id: str
    state: JobState
    pid: int
    started_at: str
    finished_at: str | None
    exit_code: int | None
    log_tail: str
    result: NotRequired[JsonValue]
    error: NotRequired[WorkerError | None]
    measured_peak_memory_gb: NotRequired[float]
    memory_metrics: NotRequired[RunMemory]
    outputs_error: NotRequired[str]


class MlxProbe(TypedDict):
    """What MLX reports in the probe subprocess; ``reason`` explains ``usable=False``."""

    usable: bool
    reason: NotRequired[str]
    version: NotRequired[str | None]
    metal_available: NotRequired[bool]
    default_device: NotRequired[str]
    device_info: NotRequired[dict[str, str | int | float | bool | None]]


class RuntimePayload(TypedDict):
    """Platform facts, installed distribution versions and the separate MLX probe.

    A package version is ``None`` when distribution metadata is absent; it does
    not prove which source files an import resolves to.
    """

    platform: str
    machine: str
    python: str
    packages: dict[str, str | None]
    mlx: MlxProbe


class RuntimeToolPayload(RuntimePayload):
    """Runtime facts plus the server root used for relative model paths."""

    root: str


class ModelPayload(TypedDict):
    """A provider declaration and its declaring distribution's metadata.

    ``run`` and ``readiness`` are ``module:attribute`` references, not calls.
    ``readiness``, ``summary``, ``package`` and ``version`` can be absent as
    metadata and are represented by ``None``.
    """

    id: str
    label: str
    task: str
    summary: str | None
    package: str | None
    version: str | None
    run: str
    readiness: str | None
    entry_point: str
    requires_weights: bool
    cohort: NotRequired[str]


class ModelListItem(ModelPayload):
    """A model declaration with its optional runtime family, or ``None``."""

    family: str | None


class DiscoveryError(TypedDict):
    """One entry point that failed discovery or declared a conflicting model id."""

    entry_point: str
    error: str


class DiscoveryWarning(TypedDict):
    """One loaded declaration with fields this MedMLX version ignores."""

    entry_point: str
    warning: str


class ModelListPayload(TypedDict):
    """Discovered models, errors and warnings; filtering hides neither."""

    models: list[ModelListItem]
    discovery_errors: list[DiscoveryError]
    discovery_warnings: list[DiscoveryWarning]


class ParameterPayload(TypedDict):
    """A runner parameter's signature and provider-authored description.

    ``default`` is ``None`` both for a required argument and for an optional
    argument whose default is ``None``; use ``required`` to distinguish them.
    An absent annotation or extracted description is also ``None``.
    """

    name: str
    required: bool
    default: JsonValue
    annotation: str | None
    description: str | None


class MemoryPeak(TypedDict):
    """Declared peak and the current version's highest measured MLX allocation peak.

    ``effective_peak_memory_gb`` is the larger of the two, the figure compared with free memory;
    ``peak_memory_source`` names which one, or is ``None`` when neither exists.
    """

    peak_memory_gb: float | None
    measured_peak_memory_gb: float | None
    effective_peak_memory_gb: float | None
    peak_memory_source: Literal["declared", "measured"] | None


class ModelDescription(ModelListItem, MemoryPeak):
    """A model declaration, signature parameters and full runner docstring.

    ``labels`` maps output label values (as strings) to names and
    ``example_arguments`` is one example runner call; both are ``None`` when the
    package does not declare them.
    """

    parameters: list[ParameterPayload]
    instructions: str | None
    labels: dict[str, str] | None
    example_arguments: dict[str, JsonValue] | None


class LabelSummary(TypedDict):
    """One label value in an output mask, its optional name and its size."""

    value: int
    name: str | None
    voxels: int
    volume_ml: float


class OutputSummary(TypedDict):
    """A NIfTI output: geometry, then label sizes for masks or value range for images.

    ``kind`` is ``"labels"`` for integer-valued masks (background 0 excluded from
    ``labels``) and ``"image"`` otherwise.
    """

    path: str
    shape: list[int]
    spacing_mm: list[float]
    dtype: str
    kind: str
    labels: list[LabelSummary]
    value_range: list[float] | None
    names_from: OutputNamesFrom


class ActiveRun(MemoryPeak):
    """A model run holding memory on this machine, from any MedMLX session."""

    job_id: str
    model_id: str
    pid: int | None
    started_at: str


class MemoryStatus(TypedDict):
    """Physical and estimated available memory, and the runs using it.

    ``available_gb`` counts free, inactive, speculative and purgeable pages
    (macOS ``vm_stat``); it is ``None`` when the host does not report it.
    """

    total_gb: float | None
    available_gb: float | None
    max_runs: int
    active_runs: list[ActiveRun]


class MemoryCheck(MemoryPeak):
    """Whether a run can be admitted now: other runs, and the effective peak versus free memory."""

    available_gb: float | None
    fits: bool | None
    other_runs: list[ActiveRun]


class DicomSeries(TypedDict):
    """One DICOM series in a folder, as SimpleITK groups it."""

    series_uid: str
    modality: str | None
    description: str | None
    files: int


class ConvertedSeries(TypedDict):
    """A DICOM series written as NIfTI, with its geometry."""

    output_path: str
    series_uid: str
    modality: str | None
    description: str | None
    shape: list[int]
    spacing_mm: list[float]


class Readiness(TypedDict):
    """A provider readiness report; ``reason`` and ``setup_hint`` are passed through as given."""

    ready: bool
    reason: object
    setup_hint: object


class PathCheck(TypedDict):
    """Existence check for one resolved input or weight path, not content validation."""

    parameter: str
    path: str
    available: bool


class WeightsCheck(TypedDict):
    """Where a run's weights come from, checked locally without downloading.

    ``source`` is ``"argument"`` for a supplied ``weights_path``, ``"cache"`` for a
    completed managed download, ``"download"`` when the run will download them
    first, ``"manual"`` when a local ``weights_path`` is required, and ``None``
    for models without learned weights.
    """

    required: bool
    source: str | None
    path: str | None


class ImageCheck(TypedDict):
    """A NIfTI input whose header was read; tensors and geometry semantics are not checked."""

    parameter: str
    path: str
    shape: list[int] | None
    valid: bool
    reason: str | None


class ModelReadiness(TypedDict):
    """Separate runtime availability from unproven input/checkpoint compatibility.

    ``ready`` is the legacy runtime boolean, identical to ``runtime_ready``.
    ``run_ready`` is ``False`` for a known blocker, ``None`` when no blocker was
    found but something could not be checked, and ``True`` when every check
    passed: runtime and Metal, argument binding, input and weight paths, weights
    supplied or cached, and every input file a readable NIfTI header. ``True``
    does not run the model or prove that weights and images are compatible.
    """

    id: str
    ready: bool
    runtime_ready: bool
    run_ready: bool | None
    reason: object
    setup_hint: object
    arguments_valid: bool | None
    path_checks: list[PathCheck]
    weights: WeightsCheck
    image_checks: list[ImageCheck]
    memory: MemoryCheck
    inputs_validated: bool
    weights_validated: bool


class ReadinessPayload(TypedDict):
    """The shared MLX probe result and each requested model's readiness report."""

    mlx_usable: bool
    models: list[ModelReadiness]


class PackagePayload(TypedDict):
    """A family requirement, its installed distribution version and model ids."""

    name: str
    version: str | None
    source: str
    models: list[str]


class FamilyPayload(TypedDict):
    """One MedMLX extra and its declared runtime packages."""

    name: str
    version: str | None
    packages: list[PackagePayload]


class PackageListPayload(TypedDict):
    """Family packages, models outside families, and discovery errors and warnings."""

    families: list[FamilyPayload]
    other_models: list[str]
    discovery_errors: list[DiscoveryError]
    discovery_warnings: list[DiscoveryWarning]


class JobListPayload(TypedDict):
    """Jobs retained by the current supervisor session."""

    jobs: list[JobPayload]


class CheckpointPayload(TypedDict):
    """Checkpoint source metadata; discovery of the runner does not qualify weights."""

    id: str
    label: str
    model_id: str
    source_url: str
    dataset: str
    input_channels: list[str]
    integration: Literal["named_adapter", "catalog_only"]
    asset_hint: str
    runner_installed: bool
    qualification: Literal["not_assessed_by_catalog"]


class CheckpointListPayload(TypedDict):
    """Checkpoint records and the catalog's qualification notice."""

    checkpoints: list[CheckpointPayload]
    notice: str
