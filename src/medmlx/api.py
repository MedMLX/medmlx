"""Shared actions for Python, command-line and MCP clients.

Model arguments and results retain each provider's declared units and geometry.
The orchestration layer does not resample images or select a substitute backend.
"""

from __future__ import annotations

import builtins
import gzip
import inspect
import math
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import NoneType, TracebackType, UnionType
from typing import Literal, Self, TypeAliasType, Union, cast, get_args, get_origin

from medmlx_core.outputs import validate_outputs
from medmlx_core.weights import cached_weights, weights_repo
from packaging.utils import canonicalize_name

from medmlx import catalog, families, memory, outputs, registry
from medmlx.jobs import Jobs
from medmlx.payloads import (
    CheckpointListPayload,
    ImageCheck,
    JobPayload,
    JsonValue,
    MemoryCheck,
    MemoryStatus,
    ModelDescription,
    ModelListItem,
    ModelListPayload,
    ModelReadiness,
    OutputNamesFrom,
    OutputSummary,
    PackageListPayload,
    PathCheck,
    ReadinessPayload,
    WeightsCheck,
)
from medmlx.probe import runtime_info
from medmlx.setup import DownloadProgress as DownloadProgress
from medmlx.setup import list_setup_plans as list_setup_plans
from medmlx.setup import setup_all as setup_all
from medmlx.setup import setup_model as setup_model
from medmlx.setup import setup_plan as setup_plan

_NIFTI_SUFFIXES = (".nii", ".nii.gz")

_SCALARS: tuple[object, ...] = (bool, int, float)

_MLX_HINT = "Use macOS arm64 (Apple Silicon) with MLX installed and Metal available."


def _canonical(name: str | None) -> str:
    return "" if name is None else canonicalize_name(name)


def _runtime_package(spec: registry.ModelSpec) -> str:
    """Attribute models to their declaring distribution or the isolated MedVLM runtime."""
    if spec.run.startswith("medmlx.adapters.medvlm:"):
        return "med-vlm-mlx"
    return _canonical(spec.package)


def list_packages() -> PackageListPayload:
    """Describe the optional runtime families, what is installed and what each declares.

    ``families`` lists the ``segment``, ``generate`` and ``vision`` extras. Each
    carries the installed ``medmlx`` version and one row per runtime requirement:
    its name, its source (a Git URL or a version specifier), its installed
    version (``None`` when absent) and the sorted ids of the models it declares.
    ``other_models`` lists discovered models whose runtime is not a member of any
    family. ``discovery_errors`` reports entry points that could not be used, and
    ``discovery_warnings`` the declaration fields this version ignores.
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
        "discovery_warnings": discovery.warnings,
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
        ``discovery_errors`` and ``discovery_warnings``, which are not filtered.
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
        "discovery_warnings": discovery.warnings,
    }


def list_checkpoints(model_id: str | None = None) -> CheckpointListPayload:
    """List checkpoint source references without downloading, loading or qualifying them.

    Source references do not register models. ``runner_installed`` reports
    discovery of the corresponding named runner only, not asset readiness.
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
        "labels": {str(value): name for value, name in spec.labels} or None,
        "example_arguments": spec.example(),
        **memory.peak_info(spec.id, spec.version, spec.peak_memory_gb),
    }


def text_arguments(model_id: str, text: Mapping[str, str]) -> dict[str, JsonValue]:
    """Convert command-line text values to the types the model's runner declares.

    Each value is converted by its parameter's evaluated annotation, ignoring
    ``None``: an annotation that admits a string or path keeps the text, ``bool``
    accepts only ``true`` or ``false``, ``int`` takes an integer and ``float`` a
    finite number. In a union of strings and scalars such as ``str | int``, text
    that would parse as one of the scalars is ambiguous and rejected; other text
    stays a string. Ambiguous values, lists, mappings and unannotated parameters
    have to be given as JSON by the caller instead.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
        ValueError: A name is not one of the runner's parameters, a value does
            not parse as its declared scalar type, or the type cannot be written
            as plain text.
    """
    spec = registry.get_model(model_id)
    declared = inspect.signature(registry.resolve(spec.run), eval_str=True).parameters
    converted: dict[str, JsonValue] = {}
    for name, value in text.items():
        parameter = declared.get(name)
        if parameter is None or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"{model_id} has no argument {name!r}; choose: {', '.join(declared)}"
            )
        converted[name] = _text_value(name, value, parameter.annotation)
    return converted


def _annotation_members(annotation: object) -> list[object]:
    """Flatten aliases, ``NewType`` and unions into their non-``None`` members."""
    if isinstance(annotation, TypeAliasType):
        return _annotation_members(annotation.__value__)
    supertype: object = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        return _annotation_members(supertype)
    if get_origin(annotation) in (Union, UnionType):
        return [member for argument in get_args(annotation)
                for member in _annotation_members(argument)]
    return [] if annotation is NoneType else [annotation]


def _is_text(member: object) -> bool:
    """Whether a plain string satisfies ``member``: ``str``, a path type or string literals."""
    if get_origin(member) is Literal:
        return all(isinstance(choice, str) for choice in get_args(member))
    kind = get_origin(member) or member
    # os.PathLike and pathlib paths define __fspath__ on the class; str does not.
    return kind is str or (isinstance(kind, type) and hasattr(kind, "__fspath__"))


def _text_value(name: str, value: str, annotation: object) -> JsonValue:
    members = _annotation_members(annotation)
    if any(_is_text(member) for member in members):
        # In a union such as ``str | int``, "3" could mean either member.
        scalars = [member for member in members if member in _SCALARS]
        if not any(_parses_as(member, value) for member in scalars):
            return value
        raise ValueError(
            f"Argument {name!r} has type {_spelled(annotation)}, so {value!r} is ambiguous; "
            f"pass it as JSON with {name}:={value} or {name}:='\"{value}\"'"
        )
    try:
        match members:
            case [builtins.bool]:
                if value not in ("true", "false"):
                    raise ValueError
                return value == "true"
            case [builtins.int]:
                return int(value)
            case [builtins.float]:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError
                return number
            case _:
                pass
    except ValueError:
        raise ValueError(
            f"Argument {name!r} expects {_spelled(annotation)}, got {value!r}"
        ) from None
    raise ValueError(
        f"Argument {name!r} has type {_spelled(annotation)}; pass it as JSON with {name}:=VALUE"
    )


def _parses_as(scalar: object, value: str) -> bool:
    try:
        _text_value("", value, scalar)
    except ValueError:
        return False
    return True


def _spelled(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return "no declared type"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("collections.abc.", "")


def _output_names(
    path: str | Path, *, job_id: str | None, model_id: str | None,
    jobs: Jobs | None, root: str | Path | None, labels: Mapping[int, str] | None,
) -> tuple[Mapping[int, str] | None, OutputNamesFrom, str | Path | None]:
    directory = root if root is not None else jobs.root if jobs is not None else None
    if job_id is not None:
        if jobs is None:
            raise ValueError("job_id requires the Jobs session that owns it (jobs=...)")
        job = jobs.get(job_id).to_payload()
        result = job.get("result")
        if job["state"] == "completed" and "outputs_error" not in job and isinstance(result, dict):
            entries = validate_outputs(result["outputs"]) if "outputs" in result else []
            file = (Path(directory or Path.cwd()) / Path(path).expanduser()).resolve()
            for entry in entries:
                candidate = (jobs.root / Path(entry["path"]).expanduser()).resolve()
                assigned = entry.get("labels")
                if candidate == file and assigned is not None:
                    return {row["value"]: row["name"] for row in assigned}, "outputs", directory
    if model_id is not None:
        declared = registry.get_model(model_id).label_names()
        return declared, "declaration" if declared else None, directory
    return labels, None, directory


def summarize_output(
    path: str | Path, *, job_id: str | None = None, model_id: str | None = None,
    jobs: Jobs | None = None, root: str | Path | None = None,
    labels: Mapping[int, str] | None = None,
) -> OutputSummary:
    """Summarize a NIfTI output using run labels, then an explicit model's labels.

    ``job_id`` requires its owning ``jobs`` session. Output paths resolve against
    that session's root. ``root`` overrides the root for the requested ``path``.
    A missing matching entry or invalid outputs list falls back to ``model_id``.
    ``names_from`` is ``outputs``, ``declaration`` or None. The existing ``labels``
    mapping can supply names without either ID; its provenance is None.

    Raises:
        LookupError: A job or model id is unknown.
        ValueError: ``job_id`` lacks a session, or the file is not NIfTI.
        FileNotFoundError: ``path`` does not exist.
        ImportError: nibabel is not installed.
    """
    names, source, directory = _output_names(
        path, job_id=job_id, model_id=model_id, jobs=jobs, root=root, labels=labels,
    )
    summary = outputs.summarize_output(path, labels=names, root=directory)
    summary["names_from"] = source
    return summary


def preview_output(
    path: str | Path, *, job_id: str | None = None, model_id: str | None = None,
    jobs: Jobs | None = None, image_path: str | Path | None = None,
    root: str | Path | None = None,
) -> bytes:
    """Render a NIfTI slice with the same naming rules as ``summarize_output``.

    Return PNG bytes with a label legend. PNG text fields contain ``labels`` as
    JSON and ``names_from`` as ``outputs``, ``declaration`` or ``null``.
    ``job_id`` requires its owning ``jobs`` session. Relative paths use that
    session's root unless ``root`` is given.

    Raises:
        LookupError: A job or model id is unknown.
        ValueError: ``job_id`` lacks a session, a file is not NIfTI, or grids differ.
        FileNotFoundError: A path does not exist.
        ImportError: nibabel or Pillow is not installed.
    """
    names, source, directory = _output_names(
        path, job_id=job_id, model_id=model_id, jobs=jobs, root=root, labels=None,
    )
    return outputs.preview_output(
        path, image_path=image_path, labels=names, names_from=source, root=directory,
    )


def model_readiness(
    model_ids: list[str] | None = None,
    *, arguments: dict[str, JsonValue] | None = None,
    root: str | Path | None = None,
) -> ReadinessPayload:
    """Check runtime availability and, optionally, one run's arguments, paths and weights.

    ``ready`` preserves the historical boolean provider/import and Metal result;
    ``runtime_ready`` names that same result explicitly. ``run_ready=False``
    identifies a known blocker. ``run_ready=True`` means every check passed: the
    runtime and Metal, argument binding, input and weight paths, weights supplied
    or already cached, and a readable NIfTI header for every input file. Otherwise
    ``run_ready=None``: no blocker was found, but something could not be checked,
    such as a non-NIfTI input or weights that the run would download first.
    No inference, downloads or output writes occur, and a true ``run_ready``
    does not prove that the weights and images are compatible.

    Args:
        model_ids: Models to check; all discovered models when omitted or empty.
        arguments: One runner's keyword arguments. Requires exactly one model id.
            Input/weight parameters ending in ``_path``, ``_paths``, ``_dir`` or
            ``_dirs`` are checked for existence; output parameters are excluded.
            NIfTI inputs (``.nii``, ``.nii.gz``) have their headers read; tensors,
            image geometry semantics and checkpoint contents are not checked.
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
            "id": spec.id, "ready": runtime["ready"],
            "runtime_ready": runtime["ready"], "run_ready": None if runtime["ready"] else False,
            "reason": runtime["reason"],
            "setup_hint": runtime["setup_hint"], "arguments_valid": None,
            "path_checks": [], "weights": _weights(spec, None, directory), "image_checks": [],
            "memory": _memory(spec), "inputs_validated": False, "weights_validated": False,
        }
        reasons = [str(row["reason"])] if row["reason"] else []
        if not mlx["usable"]:
            row.update(ready=False, runtime_ready=False, run_ready=False,
                       setup_hint=row["setup_hint"] or _MLX_HINT)
            reasons.append(f"MLX is not usable: {mlx.get('reason')}")
        if arguments is not None:
            try:
                bound = inspect.signature(registry.resolve(spec.run)).bind(**arguments)
                bound.apply_defaults()
                values = dict(bound.arguments)
                row["path_checks"] = _check_paths(values, directory)
                row["weights"] = _weights(spec, values, directory)
                row["image_checks"] = _check_images(row["path_checks"])
                row["arguments_valid"] = True
                _judge(row, reasons)
            except (TypeError, ValueError, ImportError, AttributeError) as error:
                row.update(run_ready=False, arguments_valid=False)
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


def memory_status() -> MemoryStatus:
    """Report this Mac's memory and every MedMLX model run holding it, from any session."""
    return memory.status()


def _memory(spec: registry.ModelSpec) -> MemoryCheck:
    """Check, without starting anything, whether a run of ``spec`` would be admitted now."""
    available = memory.available_bytes()
    peak = memory.peak_info(spec.id, spec.version, spec.peak_memory_gb)
    return {
        **peak,
        "available_gb": None if available is None else round(available / 1e9, 1),
        "fits": memory.fits(peak["effective_peak_memory_gb"]),
        "other_runs": memory.active_runs(),
    }


def _judge(row: ModelReadiness, reasons: list[str]) -> None:
    """Set ``run_ready`` from the run, path, weight and image checks of one bound run.

    Memory only adds a reason: a peak above the available memory does not block the run.
    """
    blockers: list[str] = []
    running = row["memory"]["other_runs"]
    if len(running) >= memory.max_runs():
        blockers.append("Another MedMLX model is running on this Mac: "
                        + ", ".join(run["model_id"] for run in running))
    if row["memory"]["fits"] is False:
        reasons.append(f"Needs about {row['memory']['effective_peak_memory_gb']} GB at peak "
                       f"and about {row['memory']['available_gb']} GB is available; "
                       "the run is not blocked")
    missing = [item["path"] for item in row["path_checks"] if not item["available"]]
    if missing:
        blockers.append("Unavailable input/weight paths: " + ", ".join(missing))
    if row["weights"]["source"] == "manual":
        blockers.append("This model needs a local weights_path folder")
    unreadable = [
        f"{item['path']} ({item['reason']})" for item in row["image_checks"] if not item["valid"]
    ]
    if unreadable:
        blockers.append("Unreadable NIfTI inputs: " + ", ".join(unreadable))
    reasons.extend(blockers)
    if blockers:
        row["run_ready"] = False
        return
    if row["run_ready"] is False:
        return
    if row["weights"]["source"] == "download":
        reasons.append(
            "Weights are not cached; the run downloads them first (prepare_model does it now)"
        )
    checked = {item["path"] for item in row["image_checks"]}
    unchecked = [
        item["path"] for item in row["path_checks"]
        if item["parameter"] != "weights_path" and item["path"] not in checked
    ]
    if unchecked:
        reasons.append("Inputs that are not NIfTI files were not checked: " + ", ".join(unchecked))
    row["run_ready"] = None if unchecked or row["weights"]["source"] == "download" else True


def _weights(
    spec: registry.ModelSpec, arguments: dict[str, object] | None, root: Path,
) -> WeightsCheck:
    """Report where a run's weights would come from, without downloading anything."""
    if not spec.requires_weights:
        return {"required": False, "source": None, "path": None}
    supplied = (arguments or {}).get("weights_path")
    if isinstance(supplied, str | Path) and str(supplied).strip():
        path = Path(supplied).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        return {"required": True, "source": "argument", "path": path.as_posix()}
    if not spec.setup_supported or weights_repo(spec.id) is None:
        return {"required": True, "source": "manual", "path": None}
    folder = cached_weights(spec.id)
    if folder is None:
        return {"required": True, "source": "download", "path": None}
    return {"required": True, "source": "cache", "path": folder.as_posix()}


def _check_images(paths: list[PathCheck]) -> list[ImageCheck]:
    """Read the header of every available NIfTI input; other formats are left unchecked."""
    checks: list[ImageCheck] = []
    for item in paths:
        if not item["available"] or not item["path"].lower().endswith(_NIFTI_SUFFIXES):
            continue
        try:
            shape = nifti_shape(item["path"])
        except (OSError, ValueError, EOFError, struct.error) as error:
            checks.append({"parameter": item["parameter"], "path": item["path"],
                           "shape": None, "valid": False, "reason": str(error)})
            continue
        checks.append({"parameter": item["parameter"], "path": item["path"],
                       "shape": shape, "valid": True, "reason": None})
    return checks


def nifti_shape(path: str | Path) -> list[int]:
    """Return a NIfTI-1 or NIfTI-2 image's dimensions from its header alone.

    Only the header is read; gzip-compressed files are decompressed as far as
    the header. Raises ``ValueError`` when the file is not a NIfTI image or
    declares impossible dimensions.
    """
    path = Path(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(540)
    for order in "<>":
        if len(header) >= 348 and struct.unpack(f"{order}i", header[:4])[0] == 348:
            dims = struct.unpack(f"{order}8h", header[40:56])
            break
        if len(header) >= 540 and struct.unpack(f"{order}i", header[:4])[0] == 540:
            dims = struct.unpack(f"{order}8q", header[16:80])
            break
    else:
        raise ValueError("not a NIfTI-1 or NIfTI-2 header")
    rank = dims[0]
    if not 1 <= rank <= 7 or any(size < 1 for size in dims[1:rank + 1]):
        raise ValueError(f"invalid NIfTI dimensions {list(dims)}")
    return [int(size) for size in dims[1:rank + 1]]


def paths_outside(arguments: dict[str, JsonValue], root: Path) -> list[str]:
    """Return argument paths, outputs included, that resolve outside ``root``.

    Path arguments are the same conventionally named parameters readiness checks
    (``*_path``, ``*_paths``, ``*_dir``, ``*_dirs``). Symlinks are resolved first.
    """
    root = root.resolve()
    outside: list[str] = []
    for name, value in arguments.items():
        if not name.endswith(("_path", "_paths", "_dir", "_dirs")) or value is None:
            continue
        items = cast(Sequence[object], value) if isinstance(value, list | tuple) else [value]
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            path = Path(item).expanduser()
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            if not path.is_relative_to(root):
                outside.append(path.as_posix())
    return outside


def start_setup(jobs: Jobs, model_id: str) -> JobPayload:
    """Start a supervised job that pre-downloads one model's weights.

    The job's result is the model's ``SetupResult``; its log tail shows download
    status lines and JSON byte events. Downloads share the managed cache and
    resume after interruption.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
        ValueError: The model needs manual assets and cannot be prepared automatically.
    """
    plan = setup_plan(model_id)
    if not plan["supported"]:
        raise ValueError(f"{model_id} does not support automatic setup: {plan['message']}")
    # Downloads do not load a model, so they do not claim memory.
    return jobs.start(
        model_id, "medmlx.setup:prepare", {"model_id": model_id}, holds_memory=False,
    ).to_payload()


def start_model(jobs: Jobs, model_id: str, arguments: dict[str, JsonValue]) -> JobPayload:
    """Bind a runner's arguments and check Metal before starting a worker.

    The arguments are only bound to the runner's signature (names and arity);
    their values are validated by the runner itself in the worker.

    Returns:
        The job payload, which may already be finished; poll ``jobs`` for completion.

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
    job = jobs.start(
        model_id, spec.run, arguments, peak_memory_gb=spec.peak_memory_gb, version=spec.version,
    )
    return job.to_payload()


class ModelRun:
    """A model run started in the background by ``start``.

    The run owns its worker session. ``wait`` returns the completed job, or
    raises with the provider's error, and releases the session; so do ``cancel``
    and ``close``. Use it as a context manager to cancel an unfinished run on exit.
    """

    def __init__(self, jobs: Jobs, job_id: str) -> None:
        self._jobs = jobs
        self.job_id = job_id

    def status(self) -> JobPayload:
        """Return the job's current state, log tail and, once finished, result or error."""
        return self._jobs.get(self.job_id).to_payload()

    def wait(self, timeout: float | None = None) -> JobPayload:
        """Block until the run finishes and return it; the run keeps going after a timeout.

        Raises:
            TimeoutError: The run is still going after ``timeout`` seconds.
            RuntimeError: The worker failed or was cancelled; the message carries
                the provider's reported error or the worker's exit code.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while (job := self._jobs.get(self.job_id)).state == "running":
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{job.model_id} is still running ({self.job_id})")
            time.sleep(0.1)
        payload = job.to_payload()
        self.close()
        if payload["state"] != "completed":
            error = payload.get("error")
            message = error["message"] if error else f"Worker exited with {payload['exit_code']}"
            raise RuntimeError(f"{payload['model_id']}: {message}")
        return payload

    def cancel(self) -> JobPayload:
        """Stop the worker and its child processes, then release the session."""
        payload = self._jobs.cancel(self.job_id).to_payload()
        self.close()
        return payload

    def close(self) -> None:
        """Release the session; an unfinished worker is cancelled."""
        self._jobs.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


def start(
    model_id: str,
    *,
    arguments: dict[str, JsonValue],
    root: str | Path | None = None,
) -> ModelRun:
    """Start a model in a supervised worker and return without waiting.

    Relative model paths resolve against ``root`` (the current directory by
    default). Call ``wait`` on the result for the completed job, or ``cancel``.

    Raises:
        NotADirectoryError: ``root`` is not an existing directory.
        LookupError: ``model_id`` is not a discovered model.
        ValueError: ``arguments`` do not match the runner's signature.
        RuntimeError: MLX Metal is unusable on this host.
    """
    directory = Path.cwd() if root is None else Path(root).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    jobs = Jobs(directory)
    try:
        submitted = start_model(jobs, model_id, arguments)
    except BaseException:
        jobs.close()
        raise
    return ModelRun(jobs, submitted["job_id"])


def start_cohort(
    model_id: str, *, preparation: dict[str, JsonValue],
    cases: Sequence[dict[str, JsonValue]], root: str | Path | None = None,
) -> ModelRun:
    """Start a finite cohort with one prepared model in one supervised worker.

    The provider must declare a ``cohort`` factory with ``run(**case)`` and
    ``close()`` hooks. ``preparation`` fixes weights and model configuration;
    ``cases`` contains 1..256 case argument objects, processed in input order.
    ``wait()`` returns ``result.cases`` entries with ``index``, ``ok`` and a
    ``result`` or ``error``; ordinary case failures do not stop later cases.
    The worker closes automatically after the last case. Its admission claim
    covers loading through close; cancellation stops its entire process group.
    Start another cohort to select different weights or configuration.
    """
    from medmlx.cohort import MAX_CASES

    directory = Path.cwd() if root is None else Path(root).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"A cohort requires 1..{MAX_CASES} cases")
    spec = registry.get_model(model_id)
    if spec.cohort is None:
        raise ValueError(f"{model_id} does not declare a prepared cohort adapter")
    try:
        inspect.signature(registry.resolve(spec.cohort)).bind(**preparation)
    except TypeError as error:
        raise ValueError(f"Invalid preparation for {model_id}: {error}") from error
    mlx = runtime_info()["mlx"]
    if not mlx["usable"]:
        raise RuntimeError(f"MLX is not usable: {mlx.get('reason')}. {_MLX_HINT}")
    jobs = Jobs(directory)
    try:
        job = jobs.start(model_id, "medmlx.cohort:run",
                         {"prepare": spec.cohort, "preparation": preparation, "cases": list(cases)},
                         peak_memory_gb=spec.peak_memory_gb, version=spec.version)
    except BaseException:
        jobs.close()
        raise
    return ModelRun(jobs, job.job_id)


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
    with start(model_id, arguments=arguments, root=root) as model_run:
        return model_run.wait()
