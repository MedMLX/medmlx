"""Preserve primitive math across supported 3D rewrite shapes and parameters."""

from contextlib import AbstractContextManager
from typing import cast

import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from medmlx_convert.boundary import as_array, as_tensor, function
from medmlx_convert.evidence import compare_output
from medmlx_convert.guidance import require_transform_parity
from medmlx_convert.rewrites import (
    FlatGroupNorm,
    FlatInstanceNorm3d,
    SeparableAvgPool3d,
    SeparableMaxPool3d,
    SeparableTrilinearUpsample,
    SeparateBias2d,
    StridedTransposedConv3d,
    ZeroInsertTransposedConv3d,
)


def compare(source: torch.nn.Module, candidate: torch.nn.Module, image: torch.Tensor) -> None:
    with torch.no_grad():
        reference = as_tensor(source.eval()(image))
        actual = as_tensor(candidate.eval()(image))
    decision = compare_output(
        "primitive_rewrite", "features", as_array(function(actual, "numpy")()),
        as_array(function(reference, "numpy")()), 1e-4,
    )
    require_transform_parity((decision,))


@settings(max_examples=24, deadline=None)
@given(size=st.integers(4, 9), stride=st.integers(1, 2), groups=st.integers(1, 2),
       bias=st.booleans(), transposed=st.booleans())
def test_bias_boundary_preserves_fp32_convolution(
    size: int, stride: int, groups: int, bias: bool, transposed: bool,
) -> None:
    """Guard source-equivalence before deliberately rounding a separate FP16 add."""
    source = (torch.nn.ConvTranspose2d(2, 4, 3, stride=stride, padding=1,
                                     output_padding=stride - 1, groups=groups, bias=bias)
              if transposed else torch.nn.Conv2d(2, 4, 3, stride=stride, padding=1,
                                                 groups=groups, bias=bias))
    image = torch.randn(1, 2, size, size + 1)
    compare(source, SeparateBias2d(source), image)


@settings(max_examples=16, deadline=None)
@given(size=st.integers(2, 5), stride=st.integers(1, 3), bias=st.booleans())
def test_nonoverlapping_transposed_convolution(size: int, stride: int, bias: bool) -> None:
    # Installed Torch's contextmanager restores CPU RNG state but lacks annotations.
    with cast(AbstractContextManager[object], function(torch.random, "fork_rng")(devices=[])):
        function(torch, "manual_seed")(0)
        source = torch.nn.ConvTranspose3d(2, 3, (stride, 2, 1), stride=(stride, 2, 1), bias=bias)
        image = torch.randn(1, 2, size, size + 1, size)
        compare(source, StridedTransposedConv3d(source), image)


@settings(max_examples=16, deadline=None)
@given(size=st.integers(2, 5), stride=st.integers(1, 2), padding=st.integers(0, 1))
def test_overlapping_transposed_convolution(size: int, stride: int, padding: int) -> None:
    with cast(AbstractContextManager[object], function(torch.random, "fork_rng")(devices=[])):
        function(torch, "manual_seed")(0)
        source = torch.nn.ConvTranspose3d(2, 3, 3, stride=stride, padding=padding,
                                          output_padding=stride - 1)
        compare(source, ZeroInsertTransposedConv3d(source), torch.randn(1, 2, size, size, size))


@settings(max_examples=16, deadline=None)
@given(size=st.integers(2, 6), affine=st.booleans())
def test_spatial_normalization(size: int, affine: bool) -> None:
    image = torch.randn((1, 4, size, size + 1, size), generator=torch.Generator().manual_seed(0))
    instance = torch.nn.InstanceNorm3d(4, affine=affine, eps=1e-3)
    group = torch.nn.GroupNorm(2, 4, affine=affine, eps=1e-3)
    compare(instance, FlatInstanceNorm3d(instance), image)
    compare(group, FlatGroupNorm(group), image)


@settings(max_examples=16, deadline=None)
@given(size=st.integers(3, 7), padding=st.integers(0, 1), stride=st.integers(1, 2))
def test_separable_max_pool(size: int, padding: int, stride: int) -> None:
    image = torch.randn((1, 2, size, size + 1, size), generator=torch.Generator().manual_seed(0))
    source = torch.nn.MaxPool3d(2, stride=stride, padding=padding)
    compare(source, SeparableMaxPool3d(source), image)


@settings(max_examples=40, deadline=None)
@given(size=st.integers(3, 7), padding=st.integers(0, 1), stride=st.integers(1, 3),
       ceil=st.booleans(), include=st.booleans(),
       divisor=st.one_of(st.none(), st.integers(1, 7)))
def test_separable_average_pool(size: int, padding: int, stride: int,
                                ceil: bool, include: bool, divisor: int | None) -> None:
    image = torch.randn((1, 2, size, size + 1, size), generator=torch.Generator().manual_seed(0))
    source = torch.nn.AvgPool3d((2, 3, 2), stride=(stride, 2, 1), padding=padding,
                              ceil_mode=ceil, count_include_pad=include, divisor_override=divisor)
    compare(source, SeparableAvgPool3d(source), image)


@settings(max_examples=40, deadline=None)
@example(size=5, align=False, scale=(1.5, 1.5, 1.5), recompute=None)
@given(size=st.integers(2, 5), align=st.booleans(),
       scale=st.tuples(*(st.sampled_from((0.5, 0.75, 1.0, 1.5, 2.0, 2.5)) for _ in range(3))),
       recompute=st.one_of(st.none(), st.booleans()))
def test_separable_trilinear_interpolation(
    size: int, align: bool, scale: tuple[float, float, float], recompute: bool | None,
) -> None:
    image = torch.randn((1, 2, size, size + 1, size), generator=torch.Generator().manual_seed(0))
    source = torch.nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=align,
                               recompute_scale_factor=recompute)
    compare(source, SeparableTrilinearUpsample(source), image)
