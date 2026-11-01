"""Checkpoint admission rejects executable pickles and keeps only host arrays."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from medmlx_core.checkpoints import mapping_from_pairs, tensor_mapping_from_payload
from medmlx_core.conversion import load_torch_checkpoint
from medmlx_core.errors import InvalidInputError

torch = pytest.importorskip("torch")


class _ExecutableCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


def test_checkpoint_loader_rejects_pickle_execution(tmp_path: Path) -> None:
    path, marker = tmp_path / "weights.pt", tmp_path / "executed"
    torch.save({"payload": _ExecutableCheckpoint(marker)}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_torch_checkpoint(path)
    assert not marker.exists()


@pytest.mark.parametrize("weights_only", [True, False])
def test_checkpoint_payload_keeps_host_arrays_only(tmp_path: Path, weights_only: bool) -> None:
    host = np.arange(12, dtype=np.float64).reshape(3, 4)
    state: dict[str, object] = {"module.weight": torch.from_numpy(host.copy())}
    if not weights_only:
        state.update(array=host, epoch=4, description="ignored")
    path = tmp_path / "synthetic.pt"
    torch.save(state, path)
    actual = tensor_mapping_from_payload(
        load_torch_checkpoint(path, weights_only=weights_only), what="synthetic"
    )
    assert list(actual) == ["module.weight"] + ([] if weights_only else ["array"])
    for value in actual.values():
        assert value.dtype == host.dtype and np.array_equal(value, host)


class _MalformedTensor:
    def numpy(self) -> list[int]:
        return [1, 2]


def test_duplicate_malformed_and_non_mapping_checkpoints_are_rejected(tmp_path: Path) -> None:
    array = np.ones((2,), dtype=np.float32)
    with pytest.raises(InvalidInputError):
        mapping_from_pairs([("same", array), ("same", array)], what="synthetic")
    with pytest.raises(InvalidInputError):
        tensor_mapping_from_payload({1: array, "1": array}, what="synthetic")
    with pytest.raises(InvalidInputError):
        tensor_mapping_from_payload({"weight": _MalformedTensor()}, what="synthetic")
    path = tmp_path / "bad.pt"
    torch.save([1, 2], path)
    with pytest.raises(InvalidInputError):
        load_torch_checkpoint(path, weights_only=False)
