"""Wire shapes produced by the server and its supervised workers."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JobState = Literal["running", "completed", "failed", "cancelled"]


class WorkerError(TypedDict):
    """A run's exception class name and message, as reported by the worker."""

    type: str
    message: str


class WorkerResponse(TypedDict):
    """The packet a worker writes: ``result`` when ``ok``, otherwise ``error``."""

    ok: bool
    result: NotRequired[JsonValue]
    error: NotRequired[WorkerError]


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


class MlxProbe(TypedDict):
    """What MLX reports in the probe subprocess; ``reason`` explains ``usable=False``."""

    usable: bool
    reason: NotRequired[str]
    version: NotRequired[str | None]
    metal_available: NotRequired[bool]
    default_device: NotRequired[str]
    device_info: NotRequired[dict[str, str | int | float | bool | None]]


class RuntimePayload(TypedDict):
    platform: str
    machine: str
    python: str
    packages: dict[str, str | None]
    mlx: MlxProbe


class RuntimeToolPayload(RuntimePayload):
    root: str


class ModelPayload(TypedDict):
    id: str
    label: str
    task: str
    summary: str | None
    package: str | None
    version: str | None
    run: str
    readiness: str | None
    entry_point: str


class ModelListItem(ModelPayload):
    family: str | None


class DiscoveryError(TypedDict):
    entry_point: str
    error: str


class ModelListPayload(TypedDict):
    models: list[ModelListItem]
    discovery_errors: list[DiscoveryError]


class ParameterPayload(TypedDict):
    name: str
    required: bool
    default: JsonValue
    annotation: str | None


class ModelDescription(ModelListItem):
    parameters: list[ParameterPayload]


class Readiness(TypedDict):
    """A provider readiness report; ``reason`` and ``setup_hint`` are passed through as given."""

    ready: bool
    reason: object
    setup_hint: object


class ModelReadiness(Readiness):
    id: str


class ReadinessPayload(TypedDict):
    mlx_usable: bool
    models: list[ModelReadiness]


class PackagePayload(TypedDict):
    name: str
    version: str | None
    source: str
    models: list[str]


class FamilyPayload(TypedDict):
    name: str
    version: str | None
    packages: list[PackagePayload]


class PackageListPayload(TypedDict):
    families: list[FamilyPayload]
    other_models: list[str]
    discovery_errors: list[DiscoveryError]


class JobListPayload(TypedDict):
    jobs: list[JobPayload]
