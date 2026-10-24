# Numerical parity policy

The tests in the table below compare floating-point results computed by
different frameworks (Torch, MONAI and NumPy on CPU against MLX on Metal), so
they use the budgets declared in `numerical_contract.json` instead of
`assert_array_equal`. Source inspection confirms the same upstream operation,
shape, layout, parameters and crop semantics in each case; what differs is FP32
reduction order, FMA use and transcendental implementations. A passing budget
does not show that every numerical error is small outside the seeded cases.

Observed Metal differences in these helper tests include absolute errors of
1.1920929e-7, 4.7683716e-7 and 3.8146973e-6, and a relative maximum of
0.00015369, so a relative-only tolerance cannot be used near zero. The budgets
preserve exact shape/dtype checks and reject all non-finite results. They are
budgets for the seeded helper tests, not a general model accuracy claim.

A host check recalculated 66 recorded outputs using the same pinned upstream
operations in float64 and rounded them to FP32. All satisfied their declared
budgets. Maximum absolute differences were 4.76837158e-6 for dot products,
4.76837158e-7 for normalization, 2.38418579e-7 for pointwise operations,
1.19209290e-7 for interpolation and 3.81469727e-6 across channel-last graphs.
This measures upstream FP32 rounding; it does not substitute for Metal testing.

| Test (relative to `tests/`) | Comparison and reason for a numerical budget |
| --- | --- |
| `test_channel_last.py::test_channel_last_seeded_graph_matches_upstream[2d]` | Compare the final graph within its cumulative budget; isolate each layer on its upstream stage input. |
| `test_channel_last.py::test_channel_last_seeded_graph_matches_upstream[3d]` | Deconvolution/InstanceNorm reductions need numerical, not bitwise, parity. |
| `test_channel_last.py::test_individual_layers_match_upstream[linear]` | FP32 dot products have different reduction/FMA order. |
| `test_channel_last.py::test_individual_layers_match_upstream[affine_ln]` | LayerNorm statistics and affine rounding differ. |
| `test_channel_last.py::test_individual_layers_match_upstream[volume]` | Compare the upstream convolution within the dot-product budget. |
| `test_channel_last.py::test_residual_blocks_match_monai[residual_same]` | Check the complete MONAI block within the graph budget. |
| `test_channel_last.py::test_residual_blocks_match_monai[residual_project]` | Check the complete projected MONAI block within the graph budget. |
| `test_equivalence.py::test_upstream_outputs[ops.silu_1]` | SiLU has framework-specific exponential/sigmoid rounding. |
| `test_equivalence.py::test_upstream_outputs[ops.conv3d_ncdhw_3]` | Biased FP32 convolution uses different accumulation order. |
| `test_equivalence.py::test_upstream_outputs[ops.conv_transpose3d_ncdhw_4]` | Transpose-convolution products/FMA differ. |
| `test_equivalence.py::test_upstream_outputs[ops.conv3d_ncdhw_6]` | Unbiased convolution needs bounded numerical parity. |
| `test_equivalence.py::test_upstream_outputs[ops.conv_transpose3d_ncdhw_7]` | Unbiased transpose convolution needs bounded numerical parity. |
| `test_equivalence.py::test_upstream_outputs[ops.group_norm_ncdhw_8]` | One-group statistics reduce in different orders. |
| `test_equivalence.py::test_upstream_outputs[ops.group_norm_ncdhw_9]` | Two-group statistics reduce in different orders. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_11]` | MONAI split/crop semantics match; convolution rounding differs. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_12]` | Same axis-0 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_13]` | Same axis-1 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_14]` | Same axis-1 transpose split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_15]` | Same axis-2 split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_16]` | Same axis-2 transpose split/crop semantics, different arithmetic. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_17]` | Unsplit convolution has the same FP32 rounding issue. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv_transpose3d_ncdhw_18]` | Unsplit transpose convolution has the same rounding issue. |
| `test_equivalence.py::test_upstream_outputs[ops.split_conv3d_ncdhw_19]` | Stride-2 split/crop matches upstream; compare numerically. |
| `test_equivalence.py::test_upstream_outputs[ops.avg_pool3d_ncdhw_23]` | Eight-value FP32 average reductions need a rounding budget. |
| `test_equivalence.py::test_upstream_outputs[ops.upsample_trilinear_ncdhw_27]` | Separable interpolation changes weighted-sum order. |
| `test_equivalence.py::test_upstream_outputs[upsample.upsample_add_ncdhw_6]` | Interpolated sums and the skip addition need a rounding budget. |
| `test_equivalence.py::test_upstream_outputs[precision.conv_1]` | Metal FP32 matrix accumulation differs from Torch CPU convolution. |
| `test_equivalence.py::test_upstream_outputs[precision.conv_2]` | The same issue applies without bias. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_3]` | FP32 Welford is compared to a NumPy float64 population reference. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_5]` | Short Welford reductions need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_6]` | Vector-tail moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_7]` | Tiled moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_8]` | Partial-tile moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_9]` | Merged-tile moments need bounded mean/variance error. |
| `test_equivalence.py::test_upstream_outputs[precision._moments_10]` | Multiple merged tiles need bounded mean/variance error. |

Runtime admission and packaging require macOS arm64 with Metal, and fixture
loading uses the single `darwin-arm64` recording directory without backend
selection.

Channel-last stage tests feed each layer its upstream stage input so its error
is isolated from earlier layers. Full 2D/3D graphs and residual blocks remain
separate composition checks. The shared numerical gate has host tests that
reject wrong values/layout/dtype/shape, NaN/Inf and malformed budgets.

Pure layout, casts, copies, nearest-neighbor replication, padding, concatenation,
ReLU, coordinates and discrete labels retain exact comparison. Constant-window
fixtures also retain exact comparison because their seeded dyadic predictor
arithmetic is exactly representable and ordered host accumulation is the contract.
