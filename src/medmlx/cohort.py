"""Finite, explicitly prepared cohorts inside one supervised model worker."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import cast

from medmlx_core.outputs import validate_outputs

from medmlx._json import require_string_keys
from medmlx.payloads import JsonValue
from medmlx.registry import resolve
from medmlx.worker import worker_error

MAX_CASES = 256


def run(
    *, prepare: str, preparation: dict[str, JsonValue], cases: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    """Prepare once, run up to 256 cases in order, and close before returning.

    The factory returns an object with ``run(**case)`` and ``close()`` methods.
    Preparation fixes the model's weights/configuration for this cohort. All
    cases are queued before launch, so no idle worker waits for more work.
    Ordinary case failures are reported individually; a worker crash or
    cancellation terminates the cohort through the supervisor's process group.
    Only finite JSON results survive a case; providers must release their
    per-case tensors before returning and release model state in ``close``.
    """
    if not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"A cohort requires 1..{MAX_CASES} cases")
    prepared = resolve(prepare)(**preparation)
    close = cast(object, getattr(prepared, "close", None))
    if not callable(close):
        raise TypeError("Cohort preparation must return an object with close()")
    try:
        runner = cast(object, getattr(prepared, "run", None))
        if not callable(runner):
            raise TypeError("Cohort preparation must return an object with run(**case)")
        call = runner
        # Reject configuration arguments or a malformed case before any case
        # publishes output. Value/geometry validation still belongs to the provider.
        signature = inspect.signature(call)
        for case in cases:
            signature.bind(**case)
        results: list[JsonValue] = []
        for index, case in enumerate(cases):
            results.append(_run_case(index, call, case))
        return {"cases": results}
    finally:
        close()


def _run_case(
    index: int, runner: Callable[..., object], case: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Return detached JSON, letting failed-case arrays and tracebacks leave scope."""
    try:
        result = runner(**case)
        if not isinstance(result, dict):
            raise TypeError("Model run must return a JSON object")
        require_string_keys(cast(dict[object, object], result))
        # Detach provider-owned mutable data and reject nonfinite values.
        value = cast(dict[str, JsonValue], json.loads(json.dumps(result, allow_nan=False)))
        item: dict[str, JsonValue] = {"index": index, "ok": True, "result": value}
        try:
            validate_outputs(value)
        except ValueError as error:
            item["outputs_error"] = str(error)
        return item
    except Exception as error:
        return {"index": index, "ok": False, "error": cast(JsonValue, worker_error(error))}
