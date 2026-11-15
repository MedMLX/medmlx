"""Ordered FP32 Metal slab blending with completed-slab host export."""

from collections.abc import Sequence
from typing import cast

import mlx.core as mx
import numpy as np

from medmlx_core.typing import ArrayEvaluator
from medmlx_core.volume_workspace import FloatArray, SlabSink

_eval = cast(ArrayEvaluator, vars(mx)["eval"])


class MlxSlabAccumulator:
    """Retain one ROI-depth of scores and counts on Metal in patch order.

    Only finalized, cropped planes cross the host boundary. FP32 multiplication,
    addition and division remain separate operations; this route does not admit
    the paired-output float64 host blending contract.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        roi: Sequence[int],
        crop: Sequence[slice],
        channels: int,
        importance: mx.array,
        sink: SlabSink,
    ) -> None:
        self.shape, self.crop = tuple(image_shape), tuple(crop)
        self.depth = roi[0]
        self.importance, self.sink = importance, sink
        self.scores = mx.zeros((channels, self.depth, *self.shape[1:]), dtype=mx.float32)
        self.counts = mx.zeros((self.depth, *self.shape[1:]), dtype=mx.float32)
        self.image: int = 0
        self.start = 0

    def _drain(self, stop: int) -> None:
        length = stop - self.start
        if not length:
            return
        lower = max(self.start, self.crop[0].start)
        upper = min(stop, self.crop[0].stop)
        if lower < upper:
            selected = (slice(lower - self.start, upper - self.start), *self.crop[1:])
            values = self.scores[(slice(None), *selected)] / self.counts[selected]
            _eval(values)
            self.sink(self.image, lower - self.crop[0].start, cast(FloatArray, np.array(values)))
        # Shift only after exporting finalized planes. Evaluation releases the
        # preceding slab graph before more windows are accumulated.
        self.scores = mx.concatenate(
            [self.scores[:, length:], mx.zeros_like(self.scores[:, :length])], axis=1
        )
        self.counts = mx.concatenate(
            [self.counts[length:], mx.zeros_like(self.counts[:length])], axis=0
        )
        _eval(self.scores, self.counts)
        self.start = stop

    def add(self, image: int, window: Sequence[slice], weighted: mx.array) -> None:
        """Blend one weighted window; callers supply lexicographically ordered patches."""
        if image != self.image:
            self._drain(self.shape[0])
            self.image, self.start = image, 0
        self._drain(window[0].start)
        selected = (slice(0, self.depth), *window[1:])
        self.scores[(slice(None), *selected)] += weighted
        self.counts[selected] += self.importance
        _eval(self.scores, self.counts)

    def finish(self) -> None:
        """Export the final image's remaining planes."""
        self._drain(self.shape[0])
