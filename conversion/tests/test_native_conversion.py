"""Differential and serialization contracts for native graph conversion on Metal."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from medmlx_core.graph.codec import dumps, loads
from medmlx_core.graph.runtime import Session

from medmlx_convert.boundary import (
    Module,
    as_array,
    as_module,
    as_tensor,
    as_tensors,
    function,
    library,
)
from medmlx_convert.native.capture import convert

_split = cast(Callable[[torch.Tensor, int, int], tuple[torch.Tensor, ...]], torch.split)


class Stateful(Module):
    """Carry immutable aliases and state through chunking, reshape and integer casting."""

    def __init__(self) -> None:
        super().__init__()
        self.offset: torch.Tensor
        self.register_buffer("offset", torch.arange(100, 106, dtype=torch.float32),
                             persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        alias = as_tensor(function(torch.ops.aten.alias, "default")(image))
        chunks = _split(alias + self.offset, 2, 1)
        return chunks[0].reshape(-1, 1, 2), chunks[2].to(torch.int32)


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 65535))
def test_converted_state_and_runtime_roundtrip(seed: int) -> None:
    model = Stateful().eval()
    image = torch.randn((1, 6), generator=torch.Generator().manual_seed(seed))
    # Tests run in the QA-owned cwd; output paths deliberately remain relative.
    folder = Path("native") / f"state-{seed}"
    if not folder.exists():
        graph = convert(model, (image,), folder).graph
        assert loads(dumps(graph)) == graph
    actual = Session.load(folder)(mx.array(image.numpy()))
    for reference, value in zip(as_tensors(model(image)), actual, strict=True):
        actual_host = as_array(function(np, "asarray")(value))
        np.testing.assert_array_equal(actual_host, reference.detach().numpy())
    assert actual[1].dtype == mx.int32


@pytest.mark.parametrize("dimensions,groups,dilation", [(2, 2, 2), (3, 1, 2)])
def test_convolution_keeps_axes_groups_and_dilation(
    dimensions: int, groups: int, dilation: int,
) -> None:
    function(torch, "manual_seed")(8)
    conv = torch.nn.Conv2d if dimensions == 2 else torch.nn.Conv3d
    model = conv(4, 6, 3, groups=groups, dilation=dilation, padding=dilation).eval()
    image = torch.randn((1, 4, *((7,) * dimensions)))
    folder = Path("native") / f"conv-{dimensions}"
    convert(model, (image,), folder)
    output, = Session.load(folder)(mx.array(image.numpy()))
    expected = as_tensor(model(image)).detach().numpy()
    np.testing.assert_allclose(np.asarray(output, dtype=np.float32), expected, atol=2e-5, rtol=2e-5)


def test_original_networks_roundtrip() -> None:
    """Exercise original upstream architectures, including 3D learned upsampling."""
    plain_unet = function(library("dynamic_network_architectures.architectures.unet"),
                          "PlainConvUNet")
    function(torch, "manual_seed")(7)
    models = (
        plain_unet(1, 3, [4, 8, 16], torch.nn.Conv3d, 3, [1, 2, 2], 2, 3, 2,
                      True, torch.nn.InstanceNorm3d, {"eps": 1e-5, "affine": True},
                      nonlin=torch.nn.LeakyReLU,
                      nonlin_kwargs={"negative_slope": 0.01, "inplace": True}),
    )
    for index, (raw, shape) in enumerate(zip(models, ((1, 1, 8, 8, 8),),
                                                strict=True)):
        model = as_module(raw)
        model.eval()
        image = torch.randn(shape)
        folder = Path("native") / f"network-{index}"
        convert(model, (image,), folder)
        output, = Session.load(folder)(mx.array(image.numpy()))
        expected = as_tensor(model(image)).detach().numpy()
        np.testing.assert_allclose(np.asarray(output, dtype=np.float32), expected,
                                   atol=2e-4, rtol=2e-4)


def test_inference_process_has_no_torch_import() -> None:
    """A saved runtime must survive an explicit source-framework import blockade."""
    folder = Path("native") / "standalone"
    convert(Stateful().eval(), (torch.ones(1, 6),), folder)
    decoder = Path("native") / "standalone-decoder"
    convert(torch.nn.ConvTranspose3d(2, 2, 2, stride=2).eval(),
             (torch.ones(1, 2, 2, 2, 2),), decoder)
    result = subprocess.run([sys.executable, "-c", """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'coreai', 'coremltools', 'medmlx_convert'}:
            raise ImportError('Forbidden inference dependency: ' + fullname)
sys.meta_path.insert(0, RejectTorch())
from pathlib import Path
import mlx.core as mx
from medmlx_core.graph.runtime import Session
output = Session.load(Path(sys.argv[1]))(mx.ones((1, 6)))
assert output[0].shape == (1, 1, 2)
volume, = Session.load(Path(sys.argv[2]))(mx.ones((1, 2, 2, 2, 2)))
assert volume.shape == (1, 2, 4, 4, 4)
assert 'torch' not in sys.modules
""", str(folder), str(decoder)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_lazy_decoder_survives_allocator_reuse() -> None:
    """Protect the shared decoder dispatch at a size that exposed the MLX defect."""
    class Decoder(Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.ConvTranspose3d(32, 32, 2, stride=2)

        def forward(self, image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            return as_tensor(self.up(image)) + skip

    function(torch, "manual_seed")(43)
    model = Decoder().eval()
    image, skip = torch.randn(1, 32, 48, 48, 48), torch.randn(1, 32, 96, 96, 96)
    folder = Path("native") / "lazy-decoder"
    convert(model, (image, skip), folder)
    expected = as_tensor(model(image, skip)).detach().numpy()
    session = Session.load(folder)
    for _ in range(3):
        output, = session(mx.array(image.numpy()), mx.array(skip.numpy()))
        np.testing.assert_allclose(np.asarray(output, dtype=np.float32), expected,
                                   atol=2e-5, rtol=2e-5)


def test_unsupported_capture_and_input_shape_fail_explicitly() -> None:
    class Unknown(Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.cos()

    folder = Path("native") / "unsupported"
    with pytest.raises(ValueError, match="Unsupported.*aten.cos"):
        convert(Unknown().eval(), (torch.ones(1, 6),), folder)
    assert not folder.exists()
    folder = Path("native") / "shape"
    convert(Stateful().eval(), (torch.ones(1, 6),), folder)
    with pytest.raises(ValueError, match="expected"):
        Session.load(folder)(mx.ones((2, 6)))


def test_input_mutation_is_rejected_before_replay() -> None:
    class Mutate(Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.add_(1)

    image = torch.ones(1, 3)
    folder = Path("native") / "mutated-input"
    with pytest.raises(ValueError, match="mutation"):
        convert(Mutate().eval(), (image,), folder)
    assert not folder.exists()
    np.testing.assert_array_equal(image.numpy(), np.ones((1, 3), dtype=np.float32))
