"""Keep model runs from exhausting this Mac's unified memory.

Every model run on the machine holds an exclusive ``flock`` on its own file in a
run registry for as long as its worker lives. The lock belongs to the worker's
open file, so the operating system releases it when the worker exits, crashes or
is killed; a dead run never blocks the next one. Starting a run takes a short
machine-wide admission lock, checks the registry, then hands the new run's lock
to its worker. Separate MCP sessions, Python callers and the CLI share it.

By default one model runs at a time per Mac. ``MEDMLX_MAX_RUNS`` raises the
limit, and ``MEDMLX_RUN_REGISTRY`` moves the registry (default: ``.runs`` in the
MedMLX cache). Memory never refuses a run: the larger of a model's declared
``peak_memory_gb`` and its highest measured peak for the current package version is
reported next to the estimated available memory (``fits``), and the caller decides.
Measurements cover MLX allocations only, excluding NumPy and preprocessing memory.
Available memory is read from ``vm_stat``: free, inactive, speculative and
purgeable pages.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import subprocess
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkstemp
from typing import Literal, cast

from medmlx.payloads import ActiveRun, MemoryPeak, MemoryStatus

_GB = 1e9
_RECLAIMABLE = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")


def _peaks_path() -> Path:
    default = Path.home() / ".cache/medmlx/peaks.json"
    return Path(os.environ.get("MEDMLX_PEAKS", default)).expanduser()


def _read_peaks(path: Path) -> dict[str, dict[str, float]]:
    """Read a complete store snapshot; missing or corrupt stores are empty."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        peaks: dict[str, dict[str, float]] = {}
        for model_id, versions in cast(dict[str, object], raw).items():
            if not isinstance(versions, dict):
                return {}
            peaks[model_id] = {}
            for version, value in cast(dict[str, object], versions).items():
                if isinstance(value, bool) or not isinstance(value, int | float):
                    return {}
                peak = float(value)
                if not math.isfinite(peak) or peak < 0:
                    return {}
                peaks[model_id][version] = peak
        return peaks
    except (OSError, ValueError, OverflowError, RecursionError):
        return {}


def peak_info(model_id: str, version: str | None, declared: float | None) -> MemoryPeak:
    """Use the larger of the declared peak and the current version's measured peak.

    A measurement from a small input can be lower than a large input needs, so it
    never lowers a declared figure.
    """
    measured = (
        None if version is None else _read_peaks(_peaks_path()).get(model_id, {}).get(version)
    )
    effective: float | None = None
    source: Literal["declared", "measured"] | None = None
    if measured is not None:
        effective, source = measured, "measured"
    if declared is not None and (effective is None or declared > effective):
        effective, source = declared, "declared"
    return {
        "peak_memory_gb": declared,
        "measured_peak_memory_gb": measured,
        "effective_peak_memory_gb": effective,
        "peak_memory_source": source,
    }


def record_peak(model_id: str, version: str | None, peak_memory_gb: float) -> None:
    """Keep the highest MLX peak per model/version with a locked, atomic update.

    The stable sibling lock is never replaced or removed. Readers see either
    complete JSON snapshot as writers replace the store under that lock.
    Unknown package versions cannot supply a reusable measurement.
    """
    if version is None:
        return
    if not math.isfinite(peak_memory_gb) or peak_memory_gb < 0:
        raise ValueError("Measured peak memory must be a finite non-negative number")
    path = _peaks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f"{path.name}.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        peaks = _read_peaks(path)
        versions = peaks.setdefault(model_id, {})
        versions[version] = max(versions.get(version, 0), peak_memory_gb)
        descriptor, name = mkstemp(dir=path.parent, prefix=f".{path.name}.")
        temporary_path = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
                json.dump(peaks, temporary, allow_nan=False)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def registry() -> Path:
    """Return the machine-wide run registry folder, creating it when needed."""
    default = Path(os.environ.get("MEDMLX_ASSET_CACHE", Path.home() / ".cache/medmlx")) / ".runs"
    folder = Path(os.environ.get("MEDMLX_RUN_REGISTRY", default)).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def max_runs() -> int:
    """Return how many model runs may hold memory at once on this machine (at least 1)."""
    try:
        return max(1, int(os.environ.get("MEDMLX_MAX_RUNS", "1")))
    except ValueError:
        return 1


def total_bytes() -> int | None:
    """Return the machine's physical memory, or ``None`` when the host does not report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def available_bytes() -> int | None:
    """Estimate reclaimable memory from ``vm_stat``; ``None`` off macOS or on failure."""
    try:
        text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    size = re.search(r"page size of (\d+) bytes", text)
    if size is None:
        return None
    pages = 0
    for name in _RECLAIMABLE:
        found = re.search(rf"^{name}:\s+(\d+)\.", text, re.MULTILINE)
        pages += int(found.group(1)) if found else 0
    return pages * int(size.group(1))


def active_runs() -> list[ActiveRun]:
    """Return the model runs holding memory on this machine, removing records of dead ones."""
    runs: list[ActiveRun] = []
    for record in sorted(registry().glob("*.json")):
        lock = record.with_suffix(".lock")
        try:
            descriptor = os.open(lock, os.O_RDWR)
        except FileNotFoundError:
            record.unlink(missing_ok=True)
            continue
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with suppress(OSError, ValueError):
                runs.append(cast(ActiveRun, json.loads(record.read_text(encoding="utf-8"))))
            continue
        finally:
            os.close(descriptor)
        # Nobody holds the lock: the worker is gone, so the record is stale.
        record.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return runs


def status() -> MemoryStatus:
    """Report memory and the MedMLX runs holding it on this machine."""
    total, available = total_bytes(), available_bytes()
    return {
        "total_gb": None if total is None else round(total / _GB, 1),
        "available_gb": None if available is None else round(available / _GB, 1),
        "max_runs": max_runs(),
        "active_runs": active_runs(),
    }


def fits(peak_memory_gb: float | None) -> bool | None:
    """Whether a peak fits in available memory; ``None`` when either is unknown."""
    available = available_bytes()
    if peak_memory_gb is None or available is None:
        return None
    return peak_memory_gb * _GB <= available


@contextmanager
def admit(
    job_id: str, model_id: str, peak_memory_gb: float | None, *, version: str | None = None,
) -> Generator[int]:
    """Register one run and yield the lock descriptor to pass to its worker.

    The caller passes the descriptor to the worker process and closes its own
    copy when this context exits; the worker then holds the lock alone. The run
    record keeps the peak figures for ``memory_status``; they are not a limit.

    Raises:
        RuntimeError: The machine already runs ``max_runs()`` models.
    """
    folder = registry()
    with (folder / ".admission").open("a+b") as admission:
        fcntl.flock(admission.fileno(), fcntl.LOCK_EX)
        runs = active_runs()
        if len(runs) >= max_runs():
            running = ", ".join(f"{run['model_id']} (pid {run['pid']})" for run in runs)
            raise RuntimeError(
                f"Another MedMLX model is already running on this Mac: {running}. "
                f"Only {max_runs()} model run(s) hold memory at once; wait for it or cancel it."
            )
        peak = peak_info(model_id, version, peak_memory_gb)
        lock = folder / f"{job_id}.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record: ActiveRun = {
                "job_id": job_id, "model_id": model_id, "pid": None,
                **peak,
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            (folder / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
            yield descriptor
        except BaseException:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
            (folder / f"{job_id}.json").unlink(missing_ok=True)
            raise
        os.close(descriptor)


def record_pid(job_id: str, pid: int) -> None:
    """Add the worker's process ID to its run record."""
    path = registry() / f"{job_id}.json"
    with suppress(OSError, ValueError):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["pid"] = pid
        path.write_text(json.dumps(record), encoding="utf-8")


def release(job_id: str) -> None:
    """Remove a finished run's record; its lock was released when the worker exited."""
    folder = registry()
    (folder / f"{job_id}.json").unlink(missing_ok=True)
    (folder / f"{job_id}.lock").unlink(missing_ok=True)


__all__ = [
    "active_runs", "admit", "available_bytes", "fits", "max_runs", "peak_info", "record_peak",
    "status", "total_bytes",
]
