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

``id``, ``label``, ``task`` and ``run`` are required non-empty strings; any
other field is an error. ``id`` uses lowercase letters, digits, ``.``, ``_``
and ``-``, and ``run`` and ``readiness`` are ``module:attribute`` references.
``run`` is called in a worker process with JSON keyword arguments and returns a
JSON-compatible mapping. ``readiness`` takes no arguments and returns a
mapping with ``ready`` and, when not ready, ``reason`` and ``setup_hint``; it
runs in the calling process, so it should be cheap. Packages need no import of
this module.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import cast

from medmlx.payloads import (
    DiscoveryError,
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
        }


@dataclass(frozen=True, slots=True)
class Discovery:
    """Valid models by id, and every entry point that could not be used."""

    models: dict[str, ModelSpec]
    errors: list[DiscoveryError]


def discover() -> Discovery:
    """Load every ``medmlx.models`` entry point; report bad ones instead of raising.

    An entry point that fails to import or validate is reported in ``errors``
    and skipped. When several entry points declare the same model id, none of
    them is served and each is reported, so a conflict cannot silently choose a
    provider. Entry points are visited in order of distribution name, then entry
    point name.
    """

    models: dict[str, ModelSpec] = {}
    errors: list[DiscoveryError] = []
    duplicates: set[str] = set()
    for entry_point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_entry_point_key):
        label = _entry_point_label(entry_point)
        try:
            spec = _spec(entry_point)
        except Exception as error:
            errors.append({"entry_point": label, "error": f"{type(error).__name__}: {error}"})
            continue
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
    return Discovery(models=models, errors=errors)


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
    """Describe the run function's parameters from its signature."""

    rows: list[ParameterPayload] = []
    for parameter in inspect.signature(resolve(spec.run)).parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        required = parameter.default is inspect.Parameter.empty
        rows.append(
            {
                "name": parameter.name,
                "required": required,
                "default": None if required else _json_value(parameter.default),
                "annotation": None
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation),
            }
        )
    return rows


def readiness(spec: ModelSpec) -> Readiness:
    """Report whether a model can run; a model without a check is ready if it imports."""

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


def _spec(entry_point: EntryPoint) -> ModelSpec:
    loaded: object = entry_point.load()
    value: object = loaded() if callable(loaded) else loaded
    if not isinstance(value, Mapping):
        raise TypeError("must name a mapping or a callable returning one")
    raw = cast(Mapping[str, object], value)
    unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
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
    return ModelSpec(
        id=model_id,
        label=cast(str, fields["label"]),
        task=cast(str, fields["task"]),
        run=cast(str, fields["run"]),
        readiness=fields["readiness"],
        summary=fields["summary"],
        package=None if distribution is None else distribution.name,
        version=None if distribution is None else distribution.version,
        entry_point=_entry_point_label(entry_point),
    )


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
