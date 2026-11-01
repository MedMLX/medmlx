"""Optional pre-downloads for models that obtain weights automatically on first use.

Supported learned-weight models return one folder as ``weights_path``. Workflows
that declare ``weights_from`` return one folder per weight argument, prepared as
the models they name. The core weights helper owns every Hub mapping. This
module does not load or qualify checkpoints.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypedDict

from medmlx_core.weights import (
    DownloadProgress,
    cached_weights,
    weights_dir,
    weights_repo,
    weights_revision,
)

from medmlx import registry
from medmlx.payloads import DiscoveryError, JsonValue


class SetupPlan(TypedDict):
    """Read-only description of an installed model's optional pre-download.

    ``cached`` reports whether ``weights_dir`` can reuse the local download.
    It is ``None`` for weight-free or manual setup. ``revision`` is the cached
    commit, and ``weights_path`` is its folder; both are ``None`` without a cache.
    A workflow lists the models it uses in ``weights_from``; it is cached when
    all of them are, and has no single ``weights_path`` or ``revision``.
    """

    model_id: str
    supported: bool
    package: str | None
    runtime_installed: bool
    source_url: str | None
    cached: bool | None
    weights_path: str | None
    revision: str | None
    #: Models whose weights a workflow uses; empty for single-folder models.
    weights_from: list[str]
    message: str


class SetupResult(TypedDict):
    """A model's prepared folder; weight-free models have empty arguments."""

    model_id: str
    arguments: dict[str, JsonValue]
    source_url: str | None
    revision: str | None


class SetupFailure(TypedDict):
    """One pre-download that failed; completed downloads remain reusable."""

    model_id: str
    error: str


class SetupBatchResult(TypedDict):
    """Every requested result and failure; partial completion is never success."""

    complete: bool
    models: list[SetupResult]
    errors: list[SetupFailure]
    discovery_errors: list[DiscoveryError]


def list_setup_plans() -> list[SetupPlan]:
    """Describe installed models without network access or model downloads."""
    return [_plan(spec) for spec in sorted(registry.discover().models.values(), key=lambda s: s.id)]


def setup_plan(model_id: str) -> SetupPlan:
    """Describe one installed model without preparing it; unknown IDs raise LookupError."""
    return _plan(registry.get_model(model_id))


def _plan(spec: registry.ModelSpec) -> SetupPlan:
    """Describe the core-mapped source, or explain why setup is manual."""
    ready = registry.readiness(spec)
    shared = [model_id for _, model_id in spec.weights_from]
    repo = (
        weights_repo(spec.id)
        if spec.requires_weights and spec.setup_supported and not shared else None
    )
    supported = spec.setup_supported and (
        all(weights_repo(model_id) is not None for model_id in shared) if shared
        else not spec.requires_weights or repo is not None
    )
    folder = None if repo is None else cached_weights(spec.id)
    cached = (
        (all(cached_weights(model_id) is not None for model_id in shared) if supported else None)
        if shared else None if repo is None else folder is not None
    )
    return {
        "model_id": spec.id,
        "supported": supported,
        "package": spec.package,
        "runtime_installed": ready["ready"],
        "source_url": None if repo is None else f"https://huggingface.co/{repo}",
        "cached": cached,
        "weights_path": None if folder is None else str(folder),
        "revision": None if folder is None else weights_revision(folder),
        "weights_from": shared,
        "message": (
            "Automatic setup is unavailable; see the runner instructions for required assets."
            if not supported
            else f"Pre-download the weights of {', '.join(shared)} for offline use; "
            "normal runs download on first use."
            if shared
            else "Pre-download model files for offline use; normal runs download on first use."
            if spec.requires_weights else "This model does not need learned weights."
        ),
    }


def setup_model(
    model_id: str, *, progress: Callable[[str], None] | None = None, refresh: bool = False,
    download_progress: Callable[[DownloadProgress], None] | None = None,
) -> SetupResult:
    """Pre-download one installed model using the same helper its runner uses.

    No inference is launched. Rerunning this resumes an interrupted download or
    reuses the completed folder offline. Runtime packages are installed with
    MedMLX's extras, not modified during a download. Explicit local folders are
    supplied as ``weights_path`` to the model run itself.

    ``progress`` receives status strings; ``download_progress`` receives byte
    events, including a final update. Unknown byte totals are ``None``.

    Raises:
        ValueError: The model requires manual asset setup.
    """
    return _prepare(
        registry.get_model(model_id), progress,
        refresh=refresh, download_progress=download_progress,
    )


def prepare(model_id: str) -> SetupResult:
    """Pre-download one model inside a supervised MedMLX job.

    The job's log tail contains status lines and JSON byte events.
    The result is the same as ``setup_model``.
    """
    return setup_model(
        model_id, progress=lambda message: print(message, flush=True),
        download_progress=lambda event: print(json.dumps(event), flush=True),
    )


def _prepare(
    spec: registry.ModelSpec, progress: Callable[[str], None] | None, *, refresh: bool,
    download_progress: Callable[[DownloadProgress], None] | None,
) -> SetupResult:
    """Prepare one supported model without inventing a source for manual assets."""
    if not spec.setup_supported:
        raise ValueError(
            f"{spec.id} does not support automatic setup; see its runner instructions for assets"
        )
    if not spec.requires_weights:
        return {"model_id": spec.id, "arguments": {}, "source_url": None, "revision": None}
    if spec.weights_from:
        # Each argument gets the same folder its model's own runner would use.
        arguments: dict[str, JsonValue] = {
            argument: str(weights_dir(model_id, progress=progress, refresh=refresh))
            for argument, model_id in spec.weights_from
        }
        return {"model_id": spec.id, "arguments": arguments, "source_url": None, "revision": None}
    repo = weights_repo(spec.id)
    if repo is None:
        raise ValueError(
            f"{spec.id} does not support automatic setup; see its runner instructions for assets"
        )
    directory = weights_dir(
        spec.id, progress=progress, refresh=refresh, download_progress=download_progress,
    )
    return {
        "model_id": spec.id,
        "arguments": {"weights_path": str(directory)},
        "source_url": f"https://huggingface.co/{repo}",
        "revision": weights_revision(directory),
    }


def setup_all(
    *, progress: Callable[[str], None] | None = None, refresh: bool = False,
    download_progress: Callable[[DownloadProgress], None] | None = None,
) -> SetupBatchResult:
    """Pre-download every installed model, continuing after ordinary model failures.

    All discovery errors and per-model failures are returned. ``complete`` is
    true only for a nonempty model list with no failures. Interrupts propagate
    immediately; completed model caches survive for the next attempt.
    Both callbacks have the same meaning as in ``setup_model``.
    """
    discovery = registry.discover()
    models = sorted(discovery.models.values(), key=lambda spec: spec.id)
    results: list[SetupResult] = []
    errors: list[SetupFailure] = []
    for index, spec in enumerate(models, start=1):
        if progress:
            progress(f"Preparing {spec.id} · {index} of {len(models)}")
        try:
            results.append(_prepare(
                spec, progress, refresh=refresh, download_progress=download_progress,
            ))
        except Exception as error:
            errors.append({"model_id": spec.id, "error": str(error) or type(error).__name__})
    return {
        "complete": bool(models) and not errors and not discovery.errors,
        "models": results,
        "errors": errors,
        "discovery_errors": discovery.errors,
    }


__all__ = ["DownloadProgress", "list_setup_plans", "setup_all", "setup_model", "setup_plan"]
