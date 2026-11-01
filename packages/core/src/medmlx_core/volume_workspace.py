"""File-backed volume intermediates and ordered host slab blending.

This module never imports MLX. Model execution remains the caller's Metal
predictor; only blending, storage and host postprocessing live here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

type FloatArray = NDArray[np.floating]
type Float32Array = NDArray[np.float32]
type SlabSink = Callable[[int, int, FloatArray], None]


class VolumeWorkspace:
    """Own temporary file-backed arrays for one complete inference/export call.

    Use as a context manager, optionally choosing a scratch directory with enough
    disk space. Arrays are valid inside that context. Files are removed on success
    or failure; no output file is overwritten. This bounds explicit resident
    intermediates, not OS page caching. Keep returned final labels outside the
    workspace. The original in-memory API remains available without this object.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        """Remember the optional scratch directory; files start on ``__enter__``."""
        self.directory = directory
        self._temporary: TemporaryDirectory[str] | None = None
        self._count = 0

    def __enter__(self) -> VolumeWorkspace:
        """Create the temporary directory, rejecting a second open."""
        if self._temporary is not None:
            raise RuntimeError("VolumeWorkspace is already open")
        self._temporary = TemporaryDirectory(prefix="medmlx-volume-", dir=self.directory)
        return self

    def __exit__(self, *exc: object) -> None:
        """Remove every file-backed array and close the workspace."""
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def empty(self, shape: Sequence[int], dtype: DTypeLike = np.float32) -> FloatArray:
        """Allocate a fresh contiguous file-backed array; every value must be written."""
        if self._temporary is None:
            raise RuntimeError("VolumeWorkspace must be open")
        path = Path(self._temporary.name) / f"volume-{self._count}.dat"
        self._count += 1
        return cast(FloatArray, np.memmap(path, mode="w+", dtype=dtype, shape=tuple(shape)))

    def map_channels(
        self,
        data: Float32Array,
        operation: Callable[[Float32Array], Float32Array],
        *,
        block_voxels: int = 65536,
    ) -> Float32Array:
        """Apply a channel-wise point operation in spatial blocks to mapped storage.

        Blocks are multiples of eight so UNesT's source softmax retains its global
        vector/scalar exponential boundary. Channel order and reduction order stay
        with ``operation``; this routine does not substitute another softmax.
        """
        if type(block_voxels) is not int or block_voxels < 8 or block_voxels % 8:
            raise ValueError("block_voxels must be a positive multiple of eight")
        output = cast(Float32Array, self.empty(data.shape))
        source, destination = data.reshape(data.shape[0], -1), output.reshape(data.shape[0], -1)
        for start in range(0, source.shape[1], block_voxels):
            stop = min(start + block_voxels, source.shape[1])
            destination[:, start:stop] = operation(source[:, start:stop])
        return output

    def float32(self, data: FloatArray, *, block_voxels: int = 65536) -> Float32Array:
        """Round wide blended logits once at export without copying the complete volume."""
        if type(block_voxels) is not int or block_voxels < 1:
            raise ValueError("block_voxels must be a positive integer")
        output = cast(Float32Array, self.empty(data.shape))
        source, destination = data.reshape(-1), output.reshape(-1)
        for start in range(0, source.size, block_voxels):
            destination[start : start + block_voxels] = source[start : start + block_voxels]
        return output


def restored_argmax(
    probabilities: Float32Array, restore: Callable[[Float32Array], Float32Array]
) -> Float32Array:
    """Restore one channel at a time and retain the first maximum on the source grid.

    All interpolation/orientation work is supplied by the model's existing
    transform. Restore probabilities before comparing; argmax before bilinear
    restoration is not equivalent. Additional working memory is independent of
    the class count; the caller still supplies the full probability array.
    """
    best = restore(probabilities[:1])[0].copy()
    labels = np.zeros(best.shape, dtype=np.float32)
    for channel in range(1, probabilities.shape[0]):
        value = restore(probabilities[channel : channel + 1])[0]
        selected = value > best
        labels[selected] = channel
        np.maximum(best, value, out=best)
    return labels


class SlabAccumulator:
    """Blend lexicographically ordered windows into a first-axis ring buffer.

    ``add`` receives already weighted (C,*ROI) predictions, in original patch
    order and predictor dtype. Adds and count sums round to ``dtype`` as in the
    dense accumulator when window order, weights and inputs match. Finalized cropped slabs go to
    ``sink(image_index, first_output_plane, values)``; they never need a complete
    class-volume accumulator. The ring depth is one ROI, independently of scan
    depth. Call ``finish`` after the final window.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        roi: Sequence[int],
        crop: Sequence[slice],
        channels: int,
        dtype: DTypeLike,
        importance: FloatArray,
        sink: SlabSink,
    ) -> None:
        """Allocate the ring buffer and reset the image and plane cursors."""
        self.shape, self.roi, self.crop = tuple(image_shape), tuple(roi), tuple(crop)
        self.importance, self.sink = importance, sink
        self.depth = self.roi[0]
        self.scores = np.zeros((channels, self.depth, *self.shape[1:]), dtype=dtype)
        self.counts = np.zeros((self.depth, *self.shape[1:]), dtype=dtype)
        self.image = 0
        self.start = 0
        self._finished = False

    def _pieces(self, origin: int, length: int) -> Iterator[tuple[slice, slice]]:
        offset = 0
        while offset < length:
            first = (origin + offset) % self.depth
            count = min(length - offset, self.depth - first)
            yield slice(offset, offset + count), slice(first, first + count)
            offset += count

    def _drain(self, stop: int) -> None:
        for source, target in self._pieces(self.start, stop - self.start):
            first = self.start + source.start
            end = self.start + source.stop
            lower, upper = max(first, self.crop[0].start), min(end, self.crop[0].stop)
            if lower < upper:
                section = slice(target.start + lower - first, target.start + upper - first)
                selected = (section, *self.crop[1:])
                values = self.scores[(slice(None), *selected)] / self.counts[selected]
                self.sink(self.image, lower - self.crop[0].start, values)
            self.scores[:, target] = 0
            self.counts[target] = 0
        self.start = stop

    def add(self, image: int, window: Sequence[slice], weighted: FloatArray) -> None:
        """Add one weighted window and flush planes no later window can touch.

        Start with image zero; the code checks that later image transitions are
        consecutive. First-axis window starts must not decrease within an image.
        The caller supplies compatible slices, patch shape and importance weights
        in the intended order.
        """
        if self._finished:
            raise RuntimeError("SlabAccumulator is finished")
        if image != self.image:
            if image != self.image + 1:
                raise ValueError("Window image indices must be consecutive")
            self._drain(self.shape[0])
            self.image, self.start = image, 0
        origin = window[0].start
        if origin < self.start:
            raise ValueError("Window starts must be nondecreasing on the streaming axis")
        self._drain(origin)
        for source, target in self._pieces(origin, self.depth):
            selected = (target, *window[1:])
            self.scores[(slice(None), *selected)] += weighted[:, source]
            self.counts[selected] += self.importance[source]

    def finish(self) -> None:
        """Flush the final image's remaining planes; repeated calls are harmless."""
        if not self._finished:
            self._drain(self.shape[0])
            self._finished = True
