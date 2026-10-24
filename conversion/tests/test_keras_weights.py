"""Protect complete, ordered and lossless Keras checkpoint admission."""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from medmlx_convert.keras_weights import load_keras_hdf5_weights


def _write(path: Path, arrays: tuple[npt.NDArray[np.float32], ...]) -> None:
    with h5py.File(str(path), "w") as file:
        # Attribute order is authoritative; the HDF5 traversal sorts these names differently.
        file.attrs["layer_names"] = ["z_conv", "a_dense"]
        for name, weights in zip(("z_conv", "a_dense"), (arrays[:2], arrays[2:]), strict=True):
            group = file.create_group(name)
            group.attrs["weight_names"] = ["kernel:0", "bias:0"]
            for key, value in zip(("kernel:0", "bias:0"), weights, strict=True):
                group.create_dataset(key, data=value)


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 65535), channels=st.integers(1, 3), filters=st.integers(1, 4))
def test_keras_hdf5_roundtrip_is_lossless_and_uses_declared_order(
    seed: int, channels: int, filters: int,
) -> None:
    rng = np.random.default_rng(seed)
    arrays = tuple(rng.normal(size=shape).astype(np.float32) for shape in (
        (3, 3, 3, channels, filters), (filters,), (filters, 2), (2,),
    ))
    path = Path("keras-roundtrip.h5")
    _write(path, arrays)
    model = torch.nn.Sequential(torch.nn.Conv3d(channels, filters, 3), torch.nn.Linear(filters, 2))
    bindings = load_keras_hdf5_weights(model, path)
    state = model.state_dict()
    restored = (state["0.weight"].numpy().transpose(2, 3, 4, 1, 0), state["0.bias"].numpy(),
                state["1.weight"].numpy().T, state["1.bias"].numpy())
    for actual, original in zip(restored, arrays, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert len(bindings) == len(state)


@settings(max_examples=8, deadline=None)
@given(extra=st.integers(1, 4), nonfinite=st.booleans())
def test_incompatible_weights_never_partially_mutate_the_model(extra: int, nonfinite: bool) -> None:
    rng = np.random.default_rng(9)
    arrays = tuple(rng.normal(size=shape).astype(np.float32) for shape in (
        (3, 3, 3, 1, 2), (2,), (2, 3 if nonfinite else 3 + extra), (3,),
    ))
    if nonfinite:
        arrays[2].flat[-1] = np.nan
    path = Path("keras-rejected.h5")
    _write(path, arrays)
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3), torch.nn.Linear(2, 3))
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="shape or nonfinite"):
        load_keras_hdf5_weights(model, path)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@given(names=st.lists(st.text(alphabet="abcdef012345/_-", max_size=12), max_size=6))
def test_malformed_hdf5_topology_fails_without_unexpected_parser_errors(
    names: list[str],
) -> None:
    path = Path("keras-topology.h5")
    with h5py.File(str(path), "w") as file:
        # HDF5 accepts these ASCII attributes; the topology references still must be rejected.
        file.attrs["layer_names"] = np.asarray(names, dtype="S")
    with pytest.raises((ValueError, KeyError)):
        load_keras_hdf5_weights(torch.nn.Linear(2, 3), path)
