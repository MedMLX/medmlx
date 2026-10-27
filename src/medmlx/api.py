"""Shared actions for Python, command-line and MCP clients.

Model arguments and results retain each provider's declared units and geometry.
The orchestration layer does not resample images or select a substitute backend.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from packaging.utils import canonicalize_name

from medmlx import catalog, families, registry
from medmlx.jobs import Jobs
from medmlx.payloads import (
    CheckpointListPayload,
    JobPayload,
    JsonValue,
    ModelDescription,
    ModelListItem,
    ModelListPayload,
    ModelReadiness,
    PackageListPayload,
    PathCheck,
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
    ``medmlx`` itself. The FDG adapter similarly belongs to ``fdg-pet-ct-mlx``.
    Every other model belongs to the distribution that
    declared its entry point (empty when the entry point has no distribution).
    """
    if spec.run.startswith("medmlx.adapters.nnunet:"):
        return "nnunet-mlx"
    if spec.run.startswith("medmlx.adapters.fdg_pet:"):
        return "fdg-pet-ct-mlx"
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


def list_checkpoints(model_id: str | None = None) -> CheckpointListPayload:
    """List checkpoint source references without downloading, loading or qualifying them.

    Catalog-only entries route to a general provider and are not registered as
    new models. ``runner_installed`` reports discovery only, not asset readiness.
    ``model_id`` filters references even when that optional runner is absent.
    """
    installed = registry.discover().models
    return {
        "checkpoints": [
            {"id": item.id, "label": item.label, "model_id": item.model_id,
             "source_url": item.source_url, "dataset": item.dataset,
             "input_channels": list(item.input_channels), "integration": item.integration,
             "asset_hint": item.asset_hint, "runner_installed": item.model_id in installed,
             "qualification": "not_assessed_by_catalog"}
            for item in catalog.CHECKPOINTS
            if model_id is None or item.model_id == model_id
        ],
        "notice": "Source metadata only. Entries do not establish staged assets, checkpoint "
                  "compatibility, numerical parity or clinical validity. "
                  "No downloads are performed.",
    }


def describe_model(model_id: str) -> ModelDescription:
    """Describe one model and the keyword arguments accepted by its runner.

    ``parameters`` is read from the runner's signature, so this imports the
    runner's module. Each row gives the parameter name, whether it is required,
    its default, annotation and provider-authored argument description.
    ``instructions`` retains the full runner docstring, including units, geometry,
    ranges and failure conditions where the provider documents them.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
    """
    spec = registry.get_model(model_id)
    members = families.family_index(families.families())
    return {
        **spec.to_payload(),
        "family": members.get(_runtime_package(spec)),
        "parameters": registry.parameters(spec),
        "instructions": inspect.getdoc(registry.resolve(spec.run)),
    }


def model_readiness(
    model_ids: list[str] | None = None,
    *, arguments: dict[str, JsonValue] | None = None,
    root: str | Path | None = None,
) -> ReadinessPayload:
    """Check runtime availability and, optionally, one run's argument names and paths.

    ``ready=False`` identifies a known blocker; ``ready=None`` means the run's
    inputs and checkpoint contents remain unverified. ``runtime_ready`` retains
    the separate provider/import and Metal result. No inference, downloads or
    output writes occur. Importable runners are never proof of compatible assets.

    Args:
        model_ids: Models to check; all discovered models when omitted or empty.
        arguments: One runner's keyword arguments. Requires exactly one model id.
            Input/weight parameters ending in ``_path``, ``_paths``, ``_dir`` or
            ``_dirs`` are checked for existence; output parameters are excluded.
            File formats, image geometry and checkpoint contents are not checked.
        root: Resolve relative paths against this directory, as runs do.

    Raises:
        LookupError: A requested id is not discovered.
        ValueError: Arguments are supplied without exactly one model id.
        NotADirectoryError: The supplied root is not an existing directory.
    """
    if arguments is not None and (not model_ids or len(model_ids) != 1):
        raise ValueError("Supply exactly one model_id when checking arguments")
    directory = Path.cwd() if root is None else Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    specs = (
        [registry.get_model(model_id) for model_id in model_ids]
        if model_ids
        else list(registry.discover().models.values())
    )
    mlx = runtime_info()["mlx"]
    rows: list[ModelReadiness] = []
    for spec in specs:
        runtime = registry.readiness(spec)
        row: ModelReadiness = {
            "id": spec.id, "ready": None if runtime["ready"] else False,
            "runtime_ready": runtime["ready"], "reason": runtime["reason"],
            "setup_hint": runtime["setup_hint"], "arguments_valid": None,
            "path_checks": [], "inputs_validated": False, "weights_validated": False,
        }
        reasons = [str(row["reason"])] if row["reason"] else []
        if not mlx["usable"]:
            row.update(ready=False, runtime_ready=False, setup_hint=row["setup_hint"] or _MLX_HINT)
            reasons.append(f"MLX is not usable: {mlx.get('reason')}")
        if arguments is not None:
            try:
                bound = inspect.signature(registry.resolve(spec.run)).bind(**arguments)
                bound.apply_defaults()
                row["path_checks"] = _check_paths(dict(bound.arguments), directory)
                row["arguments_valid"] = True
                missing = [item["path"] for item in row["path_checks"] if not item["available"]]
                if missing:
                    row["ready"] = False
                    reasons.append("Unavailable input/weight paths: " + ", ".join(missing))
            except (TypeError, ValueError, ImportError, AttributeError) as error:
                row.update(ready=False, arguments_valid=False)
                reasons.append(f"Invalid arguments: {error}")
        row["reason"] = "; ".join(reasons) or None
        rows.append(row)
    return {"mlx_usable": mlx["usable"], "models": rows}


def _check_paths(arguments: dict[str, object], root: Path) -> list[PathCheck]:
    """Check conventional input/weight paths without reading data or writing outputs."""
    checks: list[PathCheck] = []
    for name, value in arguments.items():
        if value is None or name.startswith("output"):
            continue
        plural = name.endswith(("_paths", "_dirs"))
        if not plural and not name.endswith(("_path", "_dir")):
            continue
        if plural:
            if not isinstance(value, list | tuple):
                raise ValueError(f"{name} must be a list of paths")
            values = list(cast(Sequence[object], value))
        else:
            values = [value]
        for item in values:
            if not isinstance(item, str | Path) or not str(item).strip():
                raise ValueError(f"{name} must contain non-empty paths")
            path = Path(item).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            available = path.is_dir() if name.endswith(("_dir", "_dirs")) else path.exists()
            checks.append({"parameter": name, "path": path.as_posix(), "available": available})
    return checks


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
