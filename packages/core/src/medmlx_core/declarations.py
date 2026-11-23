"""The ``MODEL`` mapping a runtime package registers in the ``medmlx.models`` entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict


class ModelDeclaration(TypedDict):
    """One model declaration, with exactly the fields MedMLX discovery accepts.

    Annotating ``MODEL: ModelDeclaration`` lets a type checker reject a
    misspelled or mistyped field that discovery would otherwise report only as
    a warning or error at runtime. ``id`` uses lowercase letters, digits, ``.``,
    ``_`` and ``-``; ``run``, ``readiness`` and ``cohort`` are
    ``module:attribute`` references. An optional field set to None counts as
    absent. ``labels`` keys are integer label values, or their decimal text.
    ``example_arguments`` must be one finite JSON object, and ``weights_from``
    maps each weight argument to the model ID whose automatic weights it uses.
    """

    id: str
    label: str
    task: str
    run: NotRequired[str | None]
    readiness: NotRequired[str | None]
    summary: NotRequired[str | None]
    cohort: NotRequired[str | None]
    requires_weights: NotRequired[bool]
    setup_supported: NotRequired[bool]
    labels: NotRequired[Mapping[int, str] | Mapping[str, str] | None]
    example_arguments: NotRequired[Mapping[str, object] | None]
    peak_memory_gb: NotRequired[float | None]
    weights_from: NotRequired[Mapping[str, str] | None]


__all__ = ["ModelDeclaration"]
