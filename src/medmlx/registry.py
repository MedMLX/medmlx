"""Discover MedMLX models that installed packages declare as entry points.

A model package declares one entry point per model in the ``medmlx.models``
group. The entry point names a mapping, or a callable returning one::

    [project.entry-points."medmlx.models"]
    totalsegmentator = "totalsegmentator_mlx.medmlx:MODEL"

    MODEL = {
        "id": "totalsegmentator",
        "label": "TotalSegmentator",
        "task": "segmentation",
        "run": "totalsegmentator_mlx.medmlx:run",
        "readiness": "totalsegmentator_mlx.medmlx:readiness",  # optional
        "summary": "Whole-body CT segmentation.",  # optional
    }

``id``, ``label``, ``task`` and ``run`` are required non-empty strings;
``readiness`` and ``summary`` are optional. ``requires_weights`` and
``setup_supported`` are optional booleans defaulting to true. ``labels``
optionally maps each integer output label to its name (``{"1": "spleen"}``), and
``example_arguments`` optionally gives one JSON object of runner arguments that
agents can adapt. ``peak_memory_gb`` optionally declares a run's peak memory in
GB. The memory guard uses the larger of this declaration and this Mac's
highest measured MLX allocation peak for the current package version; measured
peaks exclude NumPy and preprocessing memory. Runs are refused when less memory
is available. Weight-free providers set ``requires_weights`` false; providers
whose assets must be supplied by hand set ``setup_supported`` false. A workflow
that runs other models sets ``weights_from`` to a mapping from each weight
argument to the model ID whose automatic weights it uses, such as
``{"detector_weights_path": "lung-nodule"}``; setup then prepares those models.
Other fields are ignored and reported as discovery warnings, so a declaration
written for a newer MedMLX still loads.
``id`` uses lowercase letters, digits, ``.``, ``_``
and ``-``, and ``run`` and ``readiness`` are ``module:attribute`` references.
``run`` is called in a worker process with JSON keyword arguments and returns a
JSON object with string keys and finite numbers. ``readiness`` takes no arguments and returns a
mapping with ``ready`` and, when not ready, ``reason`` and ``setup_hint``; it
runs in the calling process, so it should be cheap. Packages need no import of
this module.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import cast

from medmlx.payloads import (
    DiscoveryError,
    DiscoveryWarning,
    JsonValue,
    ModelPayload,
    ParameterPayload,
    Readiness,
)

ENTRY_POINT_GROUP = "medmlx.models"

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REFERENCE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_REQUIRED = ("id", "label", "task", "run")
_OPTIONAL = ("readiness", "summary")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model a package declares, and the distribution that declares it."""

    id: str
    label: str
    task: str
    run: str
    readiness: str | None
    summary: str | None
    package: str | None
    version: str | None
    entry_point: str
    requires_weights: bool = True
    setup_supported: bool = True
    labels: tuple[tuple[int, str], ...] = ()
    example_arguments: str | None = None
    peak_memory_gb: float | None = None
    #: (run argument, model ID) pairs for workflows that reuse other models' weights.
    weights_from: tuple[tuple[str, str], ...] = ()

    def label_names(self) -> dict[int, str]:
        """Return the declared output label names by integer value."""
        return dict(self.labels)

    def example(self) -> dict[str, JsonValue] | None:
        """Return the declared example runner arguments, or ``None``."""
        if self.example_arguments is None:
            return None
        return cast(dict[str, JsonValue], json.loads(self.example_arguments))

    def to_payload(self) -> ModelPayload:
        """Return the declaration as a JSON-serializable mapping."""
        return {
            "id": self.id,
            "label": self.label,
            "task": self.task,
            "summary": self.summary,
            "package": self.package,
            "version": self.version,
            "run": self.run,
            "readiness": self.readiness,
            "entry_point": self.entry_point,
            "requires_weights": self.requires_weights,
        }


@dataclass(frozen=True, slots=True)
class Discovery:
    """Valid models by id, entry points that could not be used, and ignored fields."""

    models: dict[str, ModelSpec]
    errors: list[DiscoveryError]
    warnings: list[DiscoveryWarning] = field(default_factory=list[DiscoveryWarning])


def discover() -> Discovery:
    """Load every ``medmlx.models`` entry point; report bad ones instead of raising.

    An entry point that fails to import or validate is reported in ``errors``
    and skipped; unknown fields are reported in ``warnings`` and the model is
    kept. When several entry points declare the same model id, none of
    them is served and each is reported, so a conflict cannot silently choose a
    provider. Entry points are visited in order of distribution name, then entry
    point name.
    """

    models: dict[str, ModelSpec] = {}
    errors: list[DiscoveryError] = []
    warnings: list[DiscoveryWarning] = []
    duplicates: set[str] = set()
    for entry_point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_entry_point_key):
        label = _entry_point_label(entry_point)
        try:
            spec, unknown = _spec(entry_point)
        except Exception as error:
            errors.append({"entry_point": label, "error": f"{type(error).__name__}: {error}"})
            continue
        if unknown:
            warnings.append({
                "entry_point": label,
                "warning": f"Ignored unknown field(s): {', '.join(unknown)}",
            })
        if spec.id in models or spec.id in duplicates:
            duplicates.add(spec.id)
            errors.append(
                {"entry_point": label, "error": f"Model id {spec.id!r} is declared more than once"}
            )
            continue
        models[spec.id] = spec
    for model_id in duplicates:
        first = models.pop(model_id, None)
        if first is not None:
            errors.append(
                {
                    "entry_point": first.entry_point,
                    "error": f"Model id {model_id!r} is declared more than once",
                }
            )
    return Discovery(models=models, errors=errors, warnings=warnings)


def get_model(model_id: str) -> ModelSpec:
    """Return one discovered model, or raise ``LookupError`` naming the known ids."""

    models = discover().models
    if model_id not in models:
        known = ", ".join(sorted(models)) or "none installed"
        raise LookupError(f"Unknown MedMLX model {model_id!r}; known models: {known}")
    return models[model_id]


def resolve(reference: str) -> Callable[..., object]:
    """Import a ``module:attribute`` reference and require a callable.

    ``attribute`` may be dotted. Import and attribute errors propagate; a
    non-callable target raises ``TypeError``.
    """

    module_name, _, attribute = reference.partition(":")
    target: object = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"{reference} is not callable")
    return target


def parameters(spec: ModelSpec) -> list[ParameterPayload]:
    """Describe the signature and provider-authored argument prose.

    Google-style Args entries take priority. Otherwise, introductory paragraphs
    that explicitly name an argument in backticks supply its description.
    Descriptions are documentation, not inferred validation constraints. The full
    runner docstring remains available, including unsupported documentation formats.
    """

    runner = resolve(spec.run)
    descriptions = _argument_descriptions(inspect.getdoc(runner) or "")
    rows: list[ParameterPayload] = []
    for parameter in inspect.signature(runner).parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        required = parameter.default is inspect.Parameter.empty
        rows.append(
            {
                "name": parameter.name,
                "required": required,
                "default": None if required else _json_value(parameter.default),
                "description": descriptions.get(parameter.name),
                "annotation": None
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation),
            }
        )
    return rows


def _argument_descriptions(docstring: str) -> dict[str, str]:
    """Extract literal argument references and Args prose without inferring constraints.

    Introductory paragraphs are copied with whitespace normalized, stopping at the
    first section heading. Args entries replace this fallback, including grouped
    names wrapped across lines. Undocumented arguments retain no description.
    """
    descriptions: dict[str, str] = {}
    introduction: list[str] = []
    for line in docstring.splitlines():
        if re.fullmatch(r"[A-Za-z][\w ]*:", line):
            break
        introduction.append(line)
    for paragraph in re.split(r"\n\s*\n", "\n".join(introduction)):
        prose = " ".join(paragraph.split())
        referenced_names = {
            match.group("name") for match in re.finditer(
                r"(?<!`)(?P<ticks>`{1,2})(?P<name>[A-Za-z_]\w*)(?P=ticks)(?!`)", prose
            )
        }
        for name in referenced_names:
            previous = descriptions.get(name)
            descriptions[name] = f"{previous}\n\n{prose}" if previous else prose

    names: list[str] = []
    pending = ""
    in_args = False
    for line in docstring.splitlines():
        if line.strip() in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            names = []
            continue
        if not in_args:
            continue
        if line and not line[0].isspace():
            break
        if pending or re.match(r"^    \S", line):
            text = pending + line.strip()
            match = re.fullmatch(r"([\w, ]+)(?: \([^)]*\))?:\s*(.*)", text)
            if match:
                names = [name.strip() for name in match.group(1).split(",")]
                for name in names:
                    descriptions[name] = match.group(2).strip()
                pending = ""
                continue
            if re.fullmatch(r"(?:[A-Za-z_]\w*\s*,\s*)+", text):
                pending = text + " "
                names = []
                continue
            pending = ""
        if line.strip():
            for name in names:
                descriptions[name] += " " + line.strip()
    return descriptions


def readiness(spec: ModelSpec) -> Readiness:
    """Report the provider/import check; this does not certify run inputs or weights."""

    try:
        if spec.readiness is None:
            resolve(spec.run)
            return {"ready": True, "reason": None, "setup_hint": None}
        report = resolve(spec.readiness)()
    except Exception as error:
        return {
            "ready": False,
            "reason": f"{type(error).__name__}: {error}",
            "setup_hint": f"Install or repair the {spec.package or spec.id} package.",
        }
    values: Mapping[str, object] = (
        cast(Mapping[str, object], report) if isinstance(report, Mapping) else {}
    )
    if not isinstance(values.get("ready"), bool):
        return {
            "ready": False,
            "reason": f"{spec.readiness} must return a mapping with a boolean 'ready'",
            "setup_hint": None,
        }
    return {
        "ready": cast(bool, values["ready"]),
        "reason": values.get("reason"),
        "setup_hint": values.get("setup_hint"),
    }


def _spec(entry_point: EntryPoint) -> tuple[ModelSpec, list[str]]:
    """Validate one declaration; return it and its unknown field names."""
    loaded: object = entry_point.load()
    value: object = loaded() if callable(loaded) else loaded
    if not isinstance(value, Mapping):
        raise TypeError("must name a mapping or a callable returning one")
    raw = cast(Mapping[str, object], value)
    unknown = sorted(
        set(raw) - set(_REQUIRED) - set(_OPTIONAL)
        - {
            "requires_weights", "setup_supported", "labels", "example_arguments",
            "peak_memory_gb", "weights_from",
        }
    )
    fields: dict[str, str | None] = {}
    for key in (*_REQUIRED, *_OPTIONAL):
        item = raw.get(key)
        if item is None and key in _OPTIONAL:
            fields[key] = None
            continue
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"field {key!r} must be non-empty text")
        fields[key] = item
    model_id = cast(str, fields["id"])
    if not _ID.match(model_id):
        raise ValueError(f"id {model_id!r} must be lowercase letters, digits, '.', '_', or '-'")
    for key in ("run", "readiness"):
        reference = fields[key]
        if reference is not None and not _REFERENCE.match(reference):
            raise ValueError(f"field {key!r} must be a 'module:attribute' reference")
    distribution = entry_point.dist
    requires_weights = raw.get("requires_weights", True)
    if not isinstance(requires_weights, bool):
        raise ValueError("field 'requires_weights' must be a boolean")
    setup_supported = raw.get("setup_supported", True)
    if not isinstance(setup_supported, bool):
        raise ValueError("field 'setup_supported' must be a boolean")
    weights_from = _weights_from(raw.get("weights_from"))
    if weights_from and not requires_weights:
        raise ValueError("field 'weights_from' requires weights; remove requires_weights=False")
    spec = ModelSpec(
        labels=_labels(raw.get("labels")),
        example_arguments=_example(raw.get("example_arguments")),
        peak_memory_gb=_peak(raw.get("peak_memory_gb")),
        id=model_id,
        label=cast(str, fields["label"]),
        task=cast(str, fields["task"]),
        run=cast(str, fields["run"]),
        readiness=fields["readiness"],
        summary=fields["summary"],
        package=None if distribution is None else distribution.name,
        version=None if distribution is None else distribution.version,
        entry_point=_entry_point_label(entry_point),
        requires_weights=requires_weights,
        setup_supported=setup_supported,
        weights_from=weights_from,
    )
    return spec, unknown


def _labels(value: object) -> tuple[tuple[int, str], ...]:
    """Validate an optional ``labels`` mapping of integer values to names."""
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("field 'labels' must map integer label values to names")
    labels: list[tuple[int, str]] = []
    for key, name in cast(Mapping[object, object], value).items():
        if isinstance(key, bool) or not isinstance(key, int | str) or not str(key).isdigit():
            raise ValueError("field 'labels' keys must be non-negative integers")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("field 'labels' names must be non-empty text")
        labels.append((int(key), name))
    return tuple(sorted(labels))


def _peak(value: object) -> float | None:
    """Validate an optional positive ``peak_memory_gb`` number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ValueError("field 'peak_memory_gb' must be a positive number")
    return float(value)


def _example(value: object) -> str | None:
    """Validate optional ``example_arguments`` as one finite JSON object."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("field 'example_arguments' must be a JSON object")
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"field 'example_arguments' must be JSON: {error}") from error


def _weights_from(value: object) -> tuple[tuple[str, str], ...]:
    """Validate a ``{run argument: model ID}`` mapping, keeping declaration order."""
    if value is None:
        return ()
    if not isinstance(value, Mapping) or not value:
        raise ValueError("field 'weights_from' must be a non-empty mapping")
    pairs: list[tuple[str, str]] = []
    for argument, model_id in cast(Mapping[object, object], value).items():
        if not isinstance(argument, str) or not argument.isidentifier():
            raise ValueError("field 'weights_from' keys must be run argument names")
        if not isinstance(model_id, str) or not _ID.match(model_id):
            raise ValueError(f"field 'weights_from' has an invalid model id for {argument!r}")
        pairs.append((argument, model_id))
    return tuple(pairs)


def _entry_point_label(entry_point: EntryPoint) -> str:
    return f"{entry_point.name} = {entry_point.value}"


def _entry_point_key(entry_point: EntryPoint) -> tuple[str, str]:
    distribution = entry_point.dist
    return ("" if distribution is None else distribution.name, entry_point.name)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in cast(Sequence[object], value)]
    return str(value)
