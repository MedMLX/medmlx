"""Child-process entry point for one model run."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)

from medmlx._json import require_string_keys
from medmlx._worker_memory import WorkerMemory
from medmlx.payloads import ErrorCategory, WorkerError
from medmlx.registry import resolve

# Providers may subclass these; the category names the shared base so callers
# can branch on it without knowing each provider's exception classes.
_CATEGORIES: tuple[tuple[type[Exception], ErrorCategory], ...] = (
    (AssetNotReadyError, "AssetNotReadyError"),
    (MissingDependencyError, "MissingDependencyError"),
    (InvalidInputError, "InvalidInputError"),
    (ModelExecutionError, "ModelExecutionError"),
)


def worker_error(error: Exception) -> WorkerError:
    """Describe a runner exception for the response packet.

    ``type`` is the raised class name. ``category`` is added when the class
    derives from a shared ``medmlx_core`` error, and ``hint`` when the
    exception carries a non-empty ``hint`` string.
    """
    name = type(error).__name__
    packet: WorkerError = {"type": name, "message": str(error) or name}
    for base, category in _CATEGORIES:
        if isinstance(error, base):
            packet["category"] = category
            break
    hint = cast(object, getattr(error, "hint", None))
    if isinstance(hint, str) and hint.strip():
        packet["hint"] = hint
    return packet


def main(argv: list[str] | None = None) -> int:
    """Run one request and write its JSON-object result or execution error.

    ``REQUEST_JSON`` holds ``{"run": "module:callable", "arguments": {...}}``.
    ``RESPONSE_JSON`` receives ``{"ok": true, "result": {...}}`` with an
    optional ``measured_peak_memory_gb`` (MLX allocations only) and separate
    ``memory_metrics`` allocation/RSS counters, or
    ``{"ok": false, "error": {"type", "message"}}``; the traceback goes to
    stderr, which the supervisor captures in the job log. Returns 0 on success,
    1 when the run raised or returned something other than a finite JSON
    object, and 2 on a usage error. An unreadable or invalid-JSON request raises
    before any response is written; errors in a parsed request are reported in
    the response packet.
    """

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print("usage: python -m medmlx.worker REQUEST_JSON RESPONSE_JSON", file=sys.stderr)
        return 2
    request_path, response_path = map(Path, arguments)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    measurements = WorkerMemory()
    measurements.start()
    try:
        try:
            result = resolve(request["run"])(**request["arguments"])
            if not isinstance(result, dict):
                raise TypeError("Model run must return a JSON object")
            require_string_keys(cast(dict[object, object], result))
        finally:
            # The sampler belongs to the worker even when the runner fails.
            metrics = measurements.finish()
        response: dict[str, object] = {"ok": True, "result": result}
        response["memory_metrics"] = metrics
        peak = metrics.get("mlx_peak_bytes")
        if peak is not None:
            response["measured_peak_memory_gb"] = peak / 1e9
        text = json.dumps(response, allow_nan=False)
        exit_code = 0
    except Exception as error:
        traceback.print_exc()
        text = json.dumps({"ok": False, "error": worker_error(error)})
        exit_code = 1
    # Replace atomically so the supervisor never reads a partially written response.
    temporary = response_path.with_name(f".{response_path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, response_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
