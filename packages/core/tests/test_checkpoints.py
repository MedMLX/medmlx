from pathlib import Path

import numpy as np
import pytest
from fixture_cases import array_cases

from medmlx_core.checkpoints import (
    _maybe_array,
    load_torch_checkpoint,
    mapping_from_pairs,
    tensor_mapping_from_payload,
)
from medmlx_core.errors import InvalidInputError


def test_synthetic_weight_mapping_preserves_host_arrays() -> None:
    inputs, _ = array_cases()
    host = inputs["linear_w"].astype(np.float64)
    torch = pytest.importorskip("torch")
    state = {
        "module.weight": torch.from_numpy(host.copy()),
        "array": host,
        "epoch": 4,
        "description": "ignored",
    }
    actual = tensor_mapping_from_payload(state, what="synthetic")
    assert list(actual) == ["module.weight", "array"]
    for value in actual.values():
        assert value.dtype == host.dtype
        assert np.array_equal(value, host)
    pairs = mapping_from_pairs(list(actual.items()), what="synthetic")
    assert list(pairs) == ["module.weight", "array"]
    for key in pairs:
        assert np.array_equal(pairs[key], host)
    assert _maybe_array(host) is host
    for value in (1, 1.0, True, "text", np.float32(1), None, object()):
        assert _maybe_array(value) is None


@pytest.mark.parametrize("weights_only", [False, True])
def test_optional_torch_checkpoint_preserves_host_arrays(
    tmp_path: Path, weights_only: bool
) -> None:
    inputs, _ = array_cases()
    host = inputs["linear_w"].astype(np.float64)
    torch = pytest.importorskip("torch")
    state = {"module.weight": torch.from_numpy(host.copy())}
    if not weights_only:
        state.update(array=host, epoch=4, description="ignored")
    path = tmp_path / "synthetic.pt"
    torch.save(state, path)
    actual = tensor_mapping_from_payload(
        load_torch_checkpoint(path, weights_only=weights_only), what="synthetic"
    )
    if weights_only:
        assert list(actual) == ["module.weight"]
        assert np.array_equal(actual["module.weight"], host)
    else:
        assert list(actual) == ["module.weight", "array"]
        for value in actual.values():
            assert np.array_equal(value, host)


def test_rejects_duplicate_mapping_and_malformed_tensor() -> None:
    array = np.ones((2,), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="duplicate synthetic parameter: same"):
        mapping_from_pairs([("same", array), ("same", array)], what="synthetic")
    with pytest.raises(InvalidInputError, match="duplicate synthetic parameter: 1"):
        tensor_mapping_from_payload({1: array, "1": array}, what="synthetic")
    with pytest.raises(InvalidInputError, match="checkpoint payload must be a mapping"):
        tensor_mapping_from_payload([], what="synthetic")

    class MalformedTensor:
        def numpy(self) -> list[int]:
            return [1, 2]

    with pytest.raises(InvalidInputError, match=r"tensor numpy\(\) must return an ndarray"):
        _maybe_array(MalformedTensor())


def test_torch_loader_rejects_non_mapping_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "bad.pt"
    torch.save([1, 2], path)
    with pytest.raises(InvalidInputError, match=r"bad\.pt is not a mapping checkpoint"):
        load_torch_checkpoint(path, weights_only=False)
