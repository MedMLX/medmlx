"""Silent byte reporting through the Hub's documented ``tqdm_class`` hook."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING, Protocol, cast

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from medmlx_core.weights import DownloadProgress


class _TqdmInitializer(Protocol):
    def __init__(self, *args: object, **kwargs: object) -> None: ...


@contextmanager
def download_progress_bar(
    model_id: str, message: str, callback: Callable[[DownloadProgress], None] | None,
) -> Generator[type[tqdm[object]]]:
    """Share one throttle across a snapshot's bars and flush even on interruption."""
    lock = RLock()
    bars: list[ReportingTqdm] = []
    last_update: float | None = None

    def emit(*, final: bool = False) -> None:
        nonlocal last_update
        if callback is None:
            return
        with lock:
            now = monotonic()
            if not final and last_update is not None and now - last_update < 0.5:
                return
            total = None
            if bars and all(bar.total is not None and bar.total > 0 for bar in bars):
                total = sum(int(bar.total or 0) for bar in bars)
            last_update = now
            callback({
                "message": message,
                "model_id": model_id,
                "bytes": sum(int(bar.n) for bar in bars),
                "total_bytes": total,
            })

    class ReportingTqdm(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Disabled tqdm does not count updates, so count them ourselves.
            # Keep the library's lock, iteration and context-manager behavior.
            kwargs["disable"] = True
            name = kwargs.pop("name", "")
            # Hub 1.23 has separate transfer and reconstruction bars. Xet's
            # wire totals are estimates; completed file bytes have usable totals.
            self._reports_bytes = (
                kwargs.get("unit") == "B"
                and kwargs.get("desc") != "Downloading bytes"
                and not (isinstance(name, str) and name.endswith(".transfer"))
            )
            # tqdm's variadic constructor is an untyped third-party boundary.
            cast(_TqdmInitializer, super()).__init__(*args, **kwargs)
            if self._reports_bytes:
                with lock:
                    bars.append(self)
                    emit()

        def update(self, n: float | None = 1) -> None:
            with lock:
                self.n += n or 0
                if self._reports_bytes:
                    emit()

        def refresh(self, *_args: object, **_kwargs: object) -> None:
            if self._reports_bytes:
                emit()

    try:
        yield ReportingTqdm
    finally:
        emit(final=True)
