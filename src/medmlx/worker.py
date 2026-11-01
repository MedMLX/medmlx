"""Child-process entry point for one model run."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import cast

from medmlx.registry import resolve


def _require_string_keys(value: object) -> None:
    """Reject keys JSON would coerce, including inside arrays and nested objects."""

    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError("Model run JSON object keys must be strings")
            _require_string_keys(item)
    elif isinstance(value, list | tuple):
        for item in cast(list[object] | tuple[object, ...], value):
            _require_string_keys(item)


def _mlx_peak_gb() -> float | None:
    """Return MLX's peak allocation in GB when MLX is loaded; a failed reading is ``None``."""
    mlx = sys.modules.get("mlx.core")
    reader = getattr(mlx, "get_peak_memory", None)
    if reader is None:
        return None
    try:
        peak = float(reader()) / 1e9
    except Exception:
        # Measuring must never fail a run that succeeded.
        return None
    return peak if math.isfinite(peak) and peak >= 0 else None


def main(argv: list[str] | None = None) -> int:
    """Run one request and write its JSON-object result or execution error.

    ``REQUEST_JSON`` holds ``{"run": "module:callable", "arguments": {...}}``.
    ``RESPONSE_JSON`` receives ``{"ok": true, "result": {...}}`` with an
    optional ``measured_peak_memory_gb`` (MLX allocations only), or
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
    try:
        result = resolve(request["run"])(**request["arguments"])
        if not isinstance(result, dict):
            raise TypeError("Model run must return a JSON object")
        _require_string_keys(cast(dict[object, object], result))
        response: dict[str, object] = {"ok": True, "result": result}
        peak = _mlx_peak_gb()
        if peak is not None:
            response["measured_peak_memory_gb"] = peak
        text = json.dumps(response, allow_nan=False)
        exit_code = 0
    except Exception as error:
        traceback.print_exc()
        name = type(error).__name__
        text = json.dumps({"ok": False, "error": {"type": name, "message": str(error) or name}})
        exit_code = 1
    # Replace atomically so the supervisor never reads a partially written response.
    temporary = response_path.with_name(f".{response_path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, response_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
