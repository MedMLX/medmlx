"""Worker allocation counters and bounded process-residency sampling."""

from __future__ import annotations

import math
import resource
import sys
import threading
import time
from collections.abc import Callable
from typing import cast

import psutil

from medmlx.payloads import RunMemory

_SAMPLE_INTERVAL_SECONDS = 0.1


def _mlx_memory_bytes(name: str) -> int | None:
    """Read an available MLX counter without importing MLX or resetting its peak."""
    reader = getattr(sys.modules.get("mlx.core"), name, None)
    if not callable(reader):
        return None
    try:
        value = cast(Callable[[], object], reader)()
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return int(value) if math.isfinite(value) and value >= 0 else None
    except (RuntimeError, TypeError, ValueError, OverflowError):
        # A failed optional counter must not replace a model's result.
        return None


class WorkerMemory:
    """Keep only RSS maxima while sampling this worker and recursive descendants."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at = time.monotonic()
        self._samples = 0
        self._errors = 0
        self._worker_peak: int | None = None
        self._descendants_peak: int | None = None
        self._process_count_peak = 0

    def start(self) -> None:
        """Sample before importing the runner, then periodically until return."""
        self._started_at = time.monotonic()
        self._sample()
        self._thread.start()

    def finish(self) -> RunMemory:
        """Stop the sampler and return independent RSS and MLX counters."""
        self._stop.set()
        self._thread.join()
        self._sample()
        # Darwin's ru_maxrss is bytes; Linux reports KiB. The shipped target is
        # macOS, with Linux retained for portable host-side validation.
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
        metrics: RunMemory = {
            "scope": "runner_import_through_return",
            "worker_peak_rss_bytes": peak_rss,
            "sample_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
            "sampling_duration_seconds": time.monotonic() - self._started_at,
            "sample_count": self._samples,
            "sampling_errors": self._errors,
            "sampled_worker_rss_peak_bytes": self._worker_peak,
            "sampled_descendants_rss_sum_peak_bytes": self._descendants_peak,
            "sampled_process_count_peak": self._process_count_peak,
        }
        active = _mlx_memory_bytes("get_active_memory")
        peak = _mlx_memory_bytes("get_peak_memory")
        cache = _mlx_memory_bytes("get_cache_memory")
        if active is not None:
            metrics["mlx_active_bytes"] = active
        if peak is not None:
            metrics["mlx_peak_bytes"] = peak
        if cache is not None:
            metrics["mlx_cache_bytes"] = cache
        return metrics

    def _run(self) -> None:
        while not self._stop.wait(_SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def _sample(self) -> None:
        self._samples += 1
        try:
            worker_rss = self._process.memory_info().rss
            self._worker_peak = max(self._worker_peak or 0, worker_rss)
            children = self._process.children(recursive=True)
        except psutil.Error:
            self._errors += 1
            return
        rss = 0
        process_count = 1
        for child in children:
            try:
                rss += child.memory_info().rss
                process_count += 1
            except psutil.Error:
                # Processes can exit between traversal and their RSS reading.
                self._errors += 1
        self._descendants_peak = max(self._descendants_peak or 0, rss)
        self._process_count_peak = max(self._process_count_peak, process_count)
