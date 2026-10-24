"""Shared actions for Python, command-line and MCP clients.

Model arguments and results retain each provider's declared units and geometry.
The orchestration layer does not resample images or select a substitute backend.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

from packaging.utils import canonicalize_name

from medmlx import families, registry
from medmlx.jobs import Jobs
from medmlx.payloads import (
    JobPayload,
    JsonValue,
    ModelDescription,
    ModelListItem,
    ModelListPayload,
    ModelReadiness,
    PackageListPayload,
    ReadinessPayload,
)
from medmlx.probe import runtime_info

_MLX_HINT = "Use macOS arm64 (Apple Silicon) with MLX installed and Metal available."


def _canonical(name: str | None) -> str:
    return "" if name is None else canonicalize_name(name)


def _runtime_package(spec: registry.ModelSpec) -> str:
    """Return the canonical distribution name that supplies a model's runtime.

    The named-checkpoint adapters declared by this package run on the optional
    ``nnunet-mlx`` distribution, so they are attributed to it rather than to
    ``medmlx`` itself. Every other model belongs to the distribution that
    declared its entry point (empty when the entry point has no distribution).
    """
    if spec.run.startswith("medmlx.adapters.nnunet:"):
        return "nnunet-mlx"
    return _canonical(spec.package)


def list_packages() -> PackageListPayload:
    """Describe the optional runtime families, what is installed and what each declares.

    ``families`` lists the ``segment``, ``generate`` and ``vision`` extras. Each
    carries the installed ``medmlx`` version and one row per runtime requirement:
    its name, its source (a Git URL or a version specifier), its installed
    version (``None`` when absent) and the sorted ids of the models it declares.
    ``other_models`` lists discovered models whose runtime is not a member of any
    family. ``discovery_errors`` reports entry points that could not be used.
    """
    tree = families.families()
    discovery = registry.discover()
    declared: dict[str, list[str]] = {}
    for spec in discovery.models.values():
        declared.setdefault(_runtime_package(spec), []).append(spec.id)
    members = families.family_index(tree)
    return {
        "families": [
            {
                "name": family.name,
                "version": family.version,
                "packages": [
                    {
                        "name": package.name,
                        "version": package.version,
                        "source": package.source,
                        "models": sorted(declared.get(_canonical(package.name), [])),
                    }
                    for package in family.packages
                ],
            }
            for family in tree
        ],
        "other_models": sorted(
            spec.id for spec in discovery.models.values() if _runtime_package(spec) not in members
        ),
        "discovery_errors": discovery.errors,
    }


def list_models(task: str | None = None, family: str | None = None) -> ModelListPayload:
    """List installed model declarations and every failed declaration.

    Args:
        task: Keep only models whose declared ``task`` equals this string.
        family: Keep only models whose runtime belongs to this extra
            (``segment``, ``generate`` or ``vision``). Models outside every
            family have ``family`` of ``None`` and never match a family filter.

    Returns:
        ``models`` (one row per valid declaration, with its ``family``) and
        ``discovery_errors``; the errors are not filtered.
    """
    discovery = registry.discover()
    members = families.family_index(families.families())
    rows: list[ModelListItem] = [
        {**spec.to_payload(), "family": members.get(_runtime_package(spec))}
        for spec in discovery.models.values()
    ]
    return {
        "models": [
            row
            for row in rows
            if (task is None or row["task"] == task) and (family is None or row["family"] == family)
        ],
        "discovery_errors": discovery.errors,
    }


def describe_model(model_id: str) -> ModelDescription:
    """Describe one model and the keyword arguments accepted by its runner.

    ``parameters`` is read from the runner's signature, so this imports the
    runner's module. Each row gives the parameter name, whether it is required,
    its default (converted to a JSON value) and its annotation as text.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
    """
    spec = registry.get_model(model_id)
    members = families.family_index(families.families())
    return {
        **spec.to_payload(),
        "family": members.get(_runtime_package(spec)),
        "parameters": registry.parameters(spec),
    }


def model_readiness(model_ids: list[str] | None = None) -> ReadinessPayload:
    """Combine each provider's readiness check with the explicit Metal requirement.

    Args:
        model_ids: Models to check; all discovered models when ``None`` or empty.

    Returns:
        ``mlx_usable`` and one ``{id, ready, reason, setup_hint}`` row per model.
        A model is reported not ready whenever MLX is unusable on this host,
        even if its own check passed.

    Raises:
        LookupError: A requested id is not a discovered model.
    """
    specs = (
        [registry.get_model(model_id) for model_id in model_ids]
        if model_ids
        else list(registry.discover().models.values())
    )
    mlx = runtime_info()["mlx"]
    rows: list[ModelReadiness] = []
    for spec in specs:
        row: ModelReadiness = {"id": spec.id, **registry.readiness(spec)}
        if not mlx["usable"]:
            reasons = [str(row["reason"])] if row["reason"] else []
            row.update(
                ready=False,
                reason="; ".join([*reasons, f"MLX is not usable: {mlx.get('reason')}"]),
                setup_hint=row["setup_hint"] or _MLX_HINT,
            )
        rows.append(row)
    return {"mlx_usable": mlx["usable"], "models": rows}


def start_model(jobs: Jobs, model_id: str, arguments: dict[str, JsonValue]) -> JobPayload:
    """Validate a runner's arguments and Metal availability before starting a worker.

    The arguments are only bound to the runner's signature (names and arity);
    their values are validated by the runner itself in the worker.

    Returns:
        The job payload in the ``running`` state; poll ``jobs`` for completion.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
        ValueError: ``arguments`` do not match the runner's signature.
        RuntimeError: MLX Metal is not usable on this host.
    """
    spec = registry.get_model(model_id)
    runner = registry.resolve(spec.run)
    try:
        inspect.signature(runner).bind(**arguments)
    except TypeError as error:
        raise ValueError(f"Invalid arguments for {model_id}: {error}") from error
    mlx = runtime_info()["mlx"]
    if not mlx["usable"]:
        raise RuntimeError(f"MLX is not usable: {mlx.get('reason')}. {_MLX_HINT}")
    return jobs.start(model_id, spec.run, arguments).to_payload()


def run(
    model_id: str,
    *,
    arguments: dict[str, JsonValue],
    root: str | Path | None = None,
) -> JobPayload:
    """Run a model synchronously in a worker; cancel its process group on interruption.

    Relative model paths resolve against ``root`` (the current directory by
    default). Return the completed job and its provider result. A failed worker
    raises with its reported error instead of returning an apparent success.

    Raises:
        NotADirectoryError: ``root`` is not an existing directory.
        LookupError: ``model_id`` is not a discovered model.
        ValueError: ``arguments`` do not match the runner's signature.
        RuntimeError: MLX Metal is unusable, or the worker failed or was
            cancelled; the message carries the provider's reported error or
            the worker's exit code.
    """
    directory = Path.cwd() if root is None else Path(root).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    jobs = Jobs(directory)
    try:
        submitted = start_model(jobs, model_id, arguments)
        while (job := jobs.get(submitted["job_id"])).state == "running":
            time.sleep(0.1)
        payload = job.to_payload()
        if payload["state"] != "completed":
            error = payload.get("error")
            message = error["message"] if error else f"Worker exited with {payload['exit_code']}"
            raise RuntimeError(f"{model_id}: {message}")
        return payload
    finally:
        jobs.close()
