"""Export PyTorch models to Core AI and Core ML with source-output comparisons."""

import argparse
import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import coreai_torch
import numpy as np
import torch
from coreai.runtime import AIModel, NDArray, SpecializationOptions
from torch._ops import OperatorBase

from medmlx_convert.boundary import (
    Array,
    as_array,
    as_tensor,
    as_tensors,
    attribute,
    class_,
    function,
    item,
    library,
)
from medmlx_convert.evidence import (
    LabelRequirement,
    OutputComparison,
    Verdict,
    compare_output,
    write_evidence,
    write_runtime_fixture,
)
from medmlx_convert.guidance import (
    BuildProfile,
    GraphStage,
    graph_stages,
    native_comparison_verdict,
    require_transform_parity,
    runtime_inputs,
    transformation_comparison_verdict,
)
from medmlx_convert.precision import SourceDevice, prepare_fp16
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREML_FP16_2D, MPS_FP16, Swap, replace_modules
from medmlx_convert.runtime_benchmark import Profile

DecompositionTable = dict[OperatorBase, Callable[..., object]]
CoreMLProfile = Literal[Profile.COREML_ALL, Profile.COREML_GPU]


def _decomposition_table(value: object) -> DecompositionTable:
    if not isinstance(value, dict):
        raise TypeError(f"expected a decomposition table, got {type(value).__name__}")
    return cast(DecompositionTable, value)


def _decompose(
    program: torch.export.ExportedProgram, table: DecompositionTable
) -> torch.export.ExportedProgram:
    """``program.run_decompositions(table)``; its signature is partially untyped."""
    result = function(program, "run_decompositions")(table)
    if not isinstance(result, torch.export.ExportedProgram):
        raise TypeError(f"run_decompositions returned {type(result).__name__}")
    return result


def _numpy(t: torch.Tensor) -> Array:
    return as_array(function(t.detach().cpu().contiguous(), "numpy")(), "tensor")


def _decompositions(ops: list[object]) -> DecompositionTable:
    """PyTorch's own decompositions for ``ops`` (torch._decomp is untyped)."""
    return _decomposition_table(function(library("torch._decomp"), "get_decompositions")(ops))


# Ops coreai-torch 0.4.3 lacks, replaced with PyTorch's own decompositions before conversion.
COREAI_DECOMPOSITIONS = _decompositions([torch.ops.aten.grid_sampler_2d, torch.ops.aten.elu])


def _where_scalar_other(
    condition: torch.Tensor, x: torch.Tensor, other: float
) -> torch.Tensor:
    return torch.where(condition, x, x.new_full((), other))


def _where_scalar_self(condition: torch.Tensor, x: float, other: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, other.new_full((), x), other)


def _and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bool or y.dtype != torch.bool:
        raise NotImplementedError(
            "Core ML integer bitwise AND is unsupported; boolean inputs required"
        )
    return torch.logical_and(x, y)


def _or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bool or y.dtype != torch.bool:
        raise NotImplementedError(
            "Core ML integer bitwise OR is unsupported; boolean inputs required"
        )
    return torch.logical_or(x, y)


def _subtract(x: torch.Tensor, y: torch.Tensor, *, alpha: float = 1) -> torch.Tensor:
    """Keras uses the documented ``subtract`` alias, absent from coremltools 9.0."""
    return torch.sub(x, y, alpha=alpha)


def _alias_value(x: torch.Tensor) -> torch.Tensor:
    """Lower Torch 2.13's immutable value alias before Core ML 9 conversion.

    Core ML has no storage identity contract. The decomposed program is replayed
    against the source before conversion, including any functionalized mutations.
    """
    return x


def _conv2d_separate_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    stride: tuple[int, ...] = (1, 1), padding: tuple[int, ...] = (0, 0),
    dilation: tuple[int, ...] = (1, 1), groups: int = 1,
) -> torch.Tensor:
    """Expose bias addition so Core ML can promote all mixed operands consistently.

    Core ML Tools 9's convolution frontend promotes input/weight but omits bias,
    rejecting mixed graphs such as MedSAM2. A separate add avoids that mismatch.
    FP16 fusion/rounding can change; transformed-graph parity still must pass.
    """
    value = as_tensor(function(torch.ops.aten.convolution, "default")(
        x, weight, None, stride, padding, dilation, False, (0, 0), groups,
    ))
    return value if bias is None else value + bias.view(1, -1, 1, 1)


def _conv_transpose2d_separate_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    stride: tuple[int, ...] = (1, 1), padding: tuple[int, ...] = (0, 0),
    output_padding: tuple[int, ...] = (0, 0), groups: int = 1,
    dilation: tuple[int, ...] = (1, 1),
) -> torch.Tensor:
    """Apply the same checked bias boundary to Core ML's transposed convolution.

    Core ML Tools uses the same input/weight promotion for this operator and
    omits bias. A separate addition preserves the declared output contract;
    graph replay still rejects excessive changes in FP16 accumulation/rounding.
    """
    value = as_tensor(function(torch.ops.aten.convolution, "default")(
        x, weight, None, stride, padding, dilation, True, output_padding, groups,
    ))
    return value if bias is None else value + bias.view(1, -1, 1, 1)


# Ops coremltools 9.0 lacks, rewritten before conversion. Functional 3D nearest upsampling (which
# module swaps can't reach) and `new_ones`/`new_full` use PyTorch's own decompositions; `where`
# with a scalar branch becomes `where` with a 0-d tensor of the same dtype; tensor `&` and `|`
# become logical ops for boolean operands. Integer bitwise operations have no
# supported Core ML 9 lowering here; reject them instead of changing their meaning.
COREML_DECOMPOSITIONS: DecompositionTable = {
    **_decompositions(
        [
            torch.ops.aten.upsample_nearest3d.vec,
            torch.ops.aten.upsample_nearest3d.default,
            torch.ops.aten.new_ones,
            torch.ops.aten.new_full,
        ]
    ),
    torch.ops.aten.__and__.Tensor: _and,
    torch.ops.aten.__or__.Tensor: _or,
    torch.ops.aten.where.ScalarOther: _where_scalar_other,
    torch.ops.aten.where.ScalarSelf: _where_scalar_self,
    torch.ops.aten.subtract.Tensor: _subtract,
    torch.ops.aten.alias.default: _alias_value,
}


def _squeeze_only_singletons(program: torch.export.ExportedProgram) -> None:
    """Drop non-singleton axes from ``aten.squeeze.dims``, which PyTorch ignores.

    coreai-torch 0.4.3 lowers every listed axis to shrink_dims and fails on an axis whose
    size isn't 1, as in ``x.squeeze(2)`` on a (1, 1, 2560) tensor.
    """
    graph = program.graph_module.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.squeeze.dims:
            continue
        source, axes = node.args[0], node.args[1]
        assert isinstance(source, torch.fx.Node) and isinstance(axes, list | tuple)
        axes = cast(list[object] | tuple[object, ...], axes)
        shape = source.meta["val"].shape
        dims = [d for d in axes if isinstance(d, int) and shape[d] == 1]
        if dims:
            node.args = (source, dims)
        else:
            node.replace_all_uses_with(source)
            graph.erase_node(node)
    program.graph_module.recompile()


@dataclass(frozen=True)
class RecipeOptions:
    """Validated recipe command-line options; ``out`` is relative to the working directory."""

    out: Path
    profile: BuildProfile
    source_device: SourceDevice
    coreml_profile: CoreMLProfile


def recipe_args(name: str) -> RecipeOptions:
    """Parse a recipe's command line and prepare its output directory.

    Options: ``--out`` (repository-relative directory, default ``build/<name>`` or
    ``build/<name>-fp16``), ``--profile`` (build precision, default FP32),
    ``--source-device`` (default CPU for FP32 and MPS otherwise; FP32 requires CPU)
    and ``--coreml-profile`` (``coreml-all`` or ``coreml-gpu``).

    The output directory is created, and any ``*.verification.json`` and
    ``*.runtime.npz`` already in it are deleted so a recipe that fails before
    ``export_and_check`` cannot leave an earlier passing verdict behind. Other files
    are kept.

    Args:
        name: Recipe name, used for the default output directory.

    Returns:
        The parsed options.

    Raises:
        SystemExit: Argument parsing failed, the output path is absolute or contains
            ``..``, or the FP32 profile was given a non-CPU source device.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--profile", type=BuildProfile, choices=list(BuildProfile),
                        default=BuildProfile.FP32)
    parser.add_argument("--source-device", type=SourceDevice, choices=list(SourceDevice))
    parser.add_argument("--coreml-profile", type=Profile,
                        choices=[Profile.COREML_ALL, Profile.COREML_GPU],
                        default=Profile.COREML_ALL)
    args = parser.parse_args()
    device = args.source_device or (SourceDevice.CPU if args.profile == BuildProfile.FP32
                                     else SourceDevice.MPS)
    if args.profile == BuildProfile.FP32 and device != SourceDevice.CPU:
        parser.error("The FP32 reference profile requires --source-device cpu")
    out_dir: Path = args.out or Path("build") / (
        name if args.profile == BuildProfile.FP32 else f"{name}-fp16")
    if out_dir.is_absolute() or ".." in out_dir.parts:
        parser.error("Require a repository-relative output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Admission can fail before export_and_check is reached. Invalidate the
    # previous verdict at recipe entry, while preserving its diagnostic files.
    for pattern in ("*.verification.json", "*.runtime.npz"):
        for receipt in out_dir.glob(pattern):
            receipt.unlink()
    return RecipeOptions(out_dir, args.profile, device, args.coreml_profile)


async def _run_coreai(
    path: Path, inputs: dict[str, Array], output_names: list[str]
) -> list[Array]:
    function = (await AIModel.load(
        path, specialization_options=SpecializationOptions.default(),
    )).load_function("main")
    # Integer inputs were admitted as int32 before either runtime is invoked.
    outputs = await function(
        {name: NDArray(value) for name, value in inputs.items()}
    )
    arrays: list[Array] = []
    for name in output_names:
        value = outputs[name]
        assert isinstance(value, NDArray)
        arrays.append(as_array(value.numpy(), name))
    return arrays


def _as_tuple(result: object) -> tuple[torch.Tensor, ...]:
    return (result,) if isinstance(result, torch.Tensor) else as_tensors(result, "model output")


def _export_coreai(
    exported: torch.export.ExportedProgram,
    inputs: dict[str, torch.Tensor],
    output_names: list[str],
    path: Path,
) -> None:
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(
            exported,
            input_names=list(inputs),
            output_names=output_names,
        )
        .to_coreai()
    )
    program.save_asset(path)


def _export_coreml(
    exported: torch.export.ExportedProgram,
    inputs: dict[str, torch.Tensor],
    output_names: list[str],
    path: Path,
    profile: CoreMLProfile,
) -> None:
    ct = library("coremltools")
    tensor_type = function(ct, "TensorType")
    mlmodel = function(ct, "convert")(
        exported,
        inputs=[
            tensor_type(
                name=name,
                shape=t.shape,
                dtype=np.int32 if not t.is_floating_point() else np.float32,
            )
            for name, t in inputs.items()
        ],
        outputs=[tensor_type(name=name, dtype=np.float32) for name in output_names],
        convert_to="mlprogram",
        compute_precision=attribute(attribute(ct, "precision"), "FLOAT32"),
        minimum_deployment_target=attribute(attribute(ct, "target"), "macOS15"),
        compute_units=attribute(attribute(ct, "ComputeUnit"), profile.coreml_compute_unit),
        skip_model_load=True,
    )
    if not isinstance(mlmodel, class_(attribute(ct, "models"), "MLModel")):
        raise TypeError(f"coremltools returned {type(mlmodel).__name__}, not an MLModel")
    function(mlmodel, "save")(str(path))


def _run_coreml(path: Path, inputs: dict[str, Array], names: list[str],
                profile: CoreMLProfile) -> list[Array]:
    """Load the saved package under the explicit policy and return named outputs."""
    ct = library("coremltools")
    model = function(attribute(ct, "models"), "MLModel")(
        str(path),
        compute_units=attribute(attribute(ct, "ComputeUnit"), profile.coreml_compute_unit),
    )
    prediction = function(model, "predict")(inputs)
    return [as_array(item(prediction, name), name) for name in names]


def check_source_outputs(
    stem: Path, inputs: dict[str, torch.Tensor], actual: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...], names: list[str],
    tolerances: dict[str, float], framework: str,
    *, graph: bool = False,
    profile: BuildProfile = BuildProfile.FP32,
) -> tuple[OutputComparison, ...]:
    """Persist aligned eager outputs and reject re-authoring before native conversion.

    The caller must obtain ``reference`` before modifying the upstream model.
    FP16 graph stages use their precision-authored eager oracle on the declared
    source device. This records supplied cases, not branch or application coverage.
    """
    comparisons = tuple(compare_output(framework, name, _numpy(got), _numpy(want),
                                       tolerances[name]) for name, got, want
                        in zip(names, actual, reference, strict=True))
    comparisons = tuple(replace(out, verdict=transformation_comparison_verdict(
        out, profile, graph=graph)) for out in comparisons)
    devices = ','.join(sorted({value.device.type for value in inputs.values()}))
    reference_context = (f"PyTorch {profile} eager on {devices}"
                         if graph and profile == BuildProfile.FP16
                         else "PyTorch CPU FP32")
    write_evidence(stem, {name: _numpy(t) for name, t in inputs.items()}, list(comparisons),
                   reference=reference_context,
                   compute_policy=f"PyTorch {framework} on {devices}", build_profile=profile.value)
    require_transform_parity(comparisons, profile, graph=graph)
    return comparisons


def _capture_model(
    model: torch.nn.Module, inputs: dict[str, torch.Tensor], names: list[str],
    reference: tuple[torch.Tensor, ...], tolerances: dict[str, float],
    stem: Path, framework: str,
    profile: BuildProfile = BuildProfile.FP32,
) -> torch.export.ExportedProgram:
    """Replay each graph transformation against the independent source.

    The final replayed program is the same object supplied to the converter. This
    isolates capture/decomposition/graph-edit errors from native compiler error.
    FP32 stages retain absolute limits; FP16 stages use the declared graph policy.
    """
    with _source_attention(profile, next(iter(inputs.values())).device.type):
        program = torch.export.export(model, tuple(inputs.values()), strict=False)
    for stage in graph_stages(framework):
        if stage == GraphStage.DECOMPOSITION:
            table = COREML_DECOMPOSITIONS.copy()
            if framework == "coreml" and profile == BuildProfile.FP16:
                table[torch.ops.aten.conv2d.default] = _conv2d_separate_bias
                table[torch.ops.aten.conv_transpose2d.input] = _conv_transpose2d_separate_bias
            if framework == "coreai":
                table = _decomposition_table(function(coreai_torch, "get_decomp_table")())
                table.update(COREAI_DECOMPOSITIONS)
                if profile == BuildProfile.FP16:
                    # Remove backend-specific unused BatchNorm auxiliary tensors;
                    # Core AI's lowering otherwise assumes their CPU dtype/shape.
                    table.update(_decompositions([
                        torch.ops.aten._native_batch_norm_legit_no_training,
                        torch.ops.aten._native_batch_norm_legit,
                        torch.ops.aten.native_batch_norm,
                    ]))
            with _source_attention(profile, next(iter(inputs.values())).device.type):
                program = _decompose(program, table)
        elif stage == GraphStage.GRAPH:
            _squeeze_only_singletons(program)
        stage_stem = stem.with_name(f"{stem.name}_{framework}_{stage}")
        stage_stem.with_suffix(".graph.txt").write_text(str(program.graph) + "\n")
        with torch.no_grad(), _source_attention(profile, next(iter(inputs.values())).device.type):
            outputs = _as_tuple(program.module()(*inputs.values()))
        check_source_outputs(stage_stem, inputs, outputs, reference, names, tolerances,
                             f"{framework}_{stage}", graph=True, profile=profile)
    return program


def _prepare_model(
    model: torch.nn.Module, swaps: list[Swap], inputs: dict[str, torch.Tensor],
    names: list[str], reference: tuple[torch.Tensor, ...], tolerances: dict[str, float],
    stem: Path, framework: str,
) -> torch.nn.Module:
    """Verify a backend's prepared model before graph capture."""
    prepared = replace_modules(model, swaps).eval()
    with torch.no_grad():
        outputs = _as_tuple(prepared(*inputs.values()))
    check_source_outputs(
        stem.with_name(f"{stem.name}_{framework}_rewrite"), inputs, outputs,
        reference, names, tolerances, f"{framework}_rewrite",
    )
    return prepared


def _compare_native(
    out_stem: Path, arrays: dict[str, Array], output_names: list[str],
    reference: list[Array], tolerances: dict[str, float],
    profile: BuildProfile = BuildProfile.FP32,
    requirements: tuple[LabelRequirement, ...] = (),
    coreml_profile: CoreMLProfile = Profile.COREML_ALL,
) -> list[OutputComparison]:
    """Execute both saved assets and retain each named source-output decision."""
    results: dict[str, list[Array]] = {
        "coreai": asyncio.run(_run_coreai(out_stem.with_suffix(".aimodel"), arrays, output_names)),
        "coreml": _run_coreml(out_stem.with_suffix(".mlpackage"), arrays, output_names,
                              coreml_profile),
    }

    comparisons: list[OutputComparison] = []
    for framework, outputs in results.items():
        for name, got_raw, want_raw in zip(output_names, outputs, reference, strict=True):
            requirement = next((value for value in requirements if value.output == name), None)
            result = compare_output(framework, name, got_raw, want_raw,
                                    tolerances[name], requirement)
            result = replace(result, verdict=native_comparison_verdict(result, profile))
            comparisons.append(result)
            line = (f"{framework} {name}: max |diff| {result.max_abs_diff:.2e} "
                    f"(reference max {result.reference_max:.2e}; "
                    f"original absolute limit: {result.absolute_verdict.value}), "
                    f"raw PSNR {result.psnr_db:.2f} dB")
            psnr = result.normalized_psnr_db
            line += (f", peak-normalized PSNR {psnr:.2f} dB" if psnr is not None
                     else ", PSNR undefined (zero reference)")
            if result.label_agreement is not None:
                line += f", label agreement {result.label_agreement:.6f}"
            print(f"{line}, {result.verdict.value.upper()}")
    return comparisons


def _source_attention(profile: BuildProfile, device: str) -> AbstractContextManager[object]:
    """Keep CPU FP16 eager and graph replay on the same explicit attention kernel.

    Upstream SAM2 selects math attention inside its forward, but torch.export
    does not retain that Python context. Unrestricted replay can select CPU
    flash attention instead, introducing a kernel change at the capture stage.
    This context restores its prior state and never changes the source device.
    """
    if profile != BuildProfile.FP16 or device != "cpu":
        return nullcontext()
    attention = library("torch.nn.attention")
    context = function(attention, "sdpa_kernel")(
        attribute(attribute(attention, "SDPBackend"), "MATH"),
    )
    if not isinstance(context, AbstractContextManager):
        raise TypeError("Torch sdpa_kernel did not return a context manager")
    return cast(AbstractContextManager[object], context)


def _precision_model(
    model: torch.nn.Module, inputs: dict[str, torch.Tensor], source: tuple[torch.Tensor, ...],
    names: list[str], tolerances: dict[str, float], stem: Path, framework: str,
    device: SourceDevice,
    requirements: tuple[LabelRequirement, ...],
    fp32_norm_types: tuple[type[torch.nn.Module], ...],
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Admit intentional FP16 drift separately from graph and native compiler drift."""
    prepared = prepare_fp16(model, device, fp32_norm_types)
    moved = {name: value.to(device.value) for name, value in inputs.items()}
    with torch.no_grad(), _source_attention(BuildProfile.FP16, device.value):
        outputs = tuple(value.detach().clone() for value in _as_tuple(prepared(*moved.values())))
    comparisons = tuple(compare_output(f"{framework}_precision", name, _numpy(got),
                                      _numpy(want), tolerances[name], next((value for value
                                      in requirements if value.output == name), None))
                        for name, got, want
                        in zip(names, outputs, source, strict=True))
    decisions = tuple(replace(out, verdict=native_comparison_verdict(out, BuildProfile.FP16))
                      for out in comparisons)
    attention = "; SDPA math kernel" if device == SourceDevice.CPU else ""
    custom_norms = ", ".join(f"{value.__module__}.{value.__qualname__}"
                             for value in fp32_norm_types)
    write_evidence(stem.with_name(f"{stem.name}_{framework}_precision"),
                   {name: _numpy(value) for name, value in inputs.items()}, list(decisions),
                   compute_policy=f"PyTorch mixed FP16 eager on {device}{attention}; "
                   f"custom FP32 normalization types: {custom_norms or 'none'}",
                   build_profile=BuildProfile.FP16.value)
    if any(out.verdict != Verdict.PASS for out in decisions):
        raise ValueError(f"FP16 source precision differs beyond policy: {decisions}")
    return prepared, moved, outputs


def _invalidate_build(out_stem: Path) -> None:
    """Remove prior verdicts before any action that can reject a replacement build."""
    for suffix in (".verification.json", ".runtime.npz"):
        out_stem.with_suffix(suffix).unlink(missing_ok=True)
    for framework in ("coreai", "coreml"):
        for stage in ("rewrite", "precision", *GraphStage):
            stage_stem = out_stem.with_name(f"{out_stem.name}_{framework}_{stage}")
            stage_stem.with_suffix(".verification.json").unlink(missing_ok=True)
            stage_stem.with_suffix(".graph.txt").unlink(missing_ok=True)


def _export_formats(
    model: torch.nn.Module, inputs: dict[str, torch.Tensor], output_names: list[str],
    source_outputs: tuple[torch.Tensor, ...], tolerances: dict[str, float], out_stem: Path,
    profile: BuildProfile, source_device: SourceDevice,
    output_requirements: tuple[LabelRequirement, ...],
    coreai_swaps: list[Swap], coreml_swaps: list[Swap],
    fp32_norm_types: tuple[type[torch.nn.Module], ...],
    coreml_profile: CoreMLProfile,
) -> None:
    """Check source/precision/graph parity in order before converting each format."""
    for framework, swaps, suffix in (
        ("coreai", coreai_swaps, ".aimodel"),
        ("coreml", coreml_swaps, ".mlpackage"),
    ):
        if profile == BuildProfile.FP16 and framework == "coreml":
            swaps = [*COREML_FP16_2D, *swaps]
        if profile == BuildProfile.FP16 and source_device == SourceDevice.MPS:
            swaps = [*MPS_FP16, *swaps]
        prepared = _prepare_model(
            model, swaps, inputs, output_names, source_outputs, tolerances, out_stem, framework,
        )
        graph_inputs, graph_reference = inputs, source_outputs
        if profile == BuildProfile.FP16:
            prepared, graph_inputs, graph_reference = _precision_model(
                prepared, inputs, source_outputs, output_names, tolerances, out_stem,
                framework, source_device, output_requirements, fp32_norm_types,
            )
        program = _capture_model(
            prepared, graph_inputs, output_names, graph_reference, tolerances, out_stem, framework,
            profile,
        )
        if framework == "coreml":
            _export_coreml(program, inputs, output_names, out_stem.with_suffix(suffix),
                           coreml_profile)
        else:
            _export_coreai(program, inputs, output_names, out_stem.with_suffix(suffix))
        del prepared, program


def _require_cpu_oracle(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> None:
    """Reject mislabeled reference execution before authoring either native build."""
    parameters = tuple(model.parameters())
    tensors = (*inputs.values(), *parameters, *model.buffers())
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError(
            "The independent CPU FP32 oracle requires CPU inputs, parameters and buffers")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in tensors):
        raise ValueError("The independent CPU FP32 oracle requires FP32 floating tensors")


def export_and_check(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    output_names: list[str],
    out_stem: Path,
    tolerance: float | dict[str, float],
    *,
    source: BuildSource,
    profile: BuildProfile = BuildProfile.FP32,
    source_device: SourceDevice = SourceDevice.CPU,
    output_requirements: tuple[LabelRequirement, ...] = (),
    fp32_norm_types: tuple[type[torch.nn.Module], ...] = (),
    coreml_profile: CoreMLProfile = Profile.COREML_ALL,
    coreai_swaps: list[Swap] | None = None,
    coreml_swaps: list[Swap] | None = None,
) -> bool:
    """Export both formats and retain source, graph and native numerical decisions.

    ``model`` must supply CPU inputs/parameters/buffers with FP32 floating tensors for the
    independent oracle; wrong-device/precision references raise. It takes inputs
    positionally in mapping order and returns tensors matching
    ``output_names``. Format-specific module swaps are applied to evaluated copies
    before each graph capture. Return false for a numerical
    failure; conversion/runtime errors propagate without a passing numerical report.

    FP16 keeps FP32 absolute errors as diagnostics, requires raw PSNR >50 dB
    versus the CPU FP32 oracle, and replays each graph stage at >70 dB against
    its precision-authored eager model. Declared label budgets apply to source
    precision and native outputs. Passing admits these tensors, not deployment.
    ``fp32_norm_types`` explicitly extends standard Torch normalization handling
    to exact custom module types; original FP32 weights are retained on the copy.
    ``coreml_profile`` explicitly selects ALL or CPU_AND_GPU for conversion and
    saved-package execution. It never retries a failed runtime with another policy.
    """
    _invalidate_build(out_stem)
    _require_cpu_oracle(model, inputs)
    if coreml_profile not in (Profile.COREML_ALL, Profile.COREML_GPU):
        raise ValueError(f"Unsupported Core ML build runtime policy: {coreml_profile}")
    if profile == BuildProfile.FP32 and source_device != SourceDevice.CPU:
        raise ValueError("The FP32 reference profile requires the CPU source device")
    tolerances: dict[str, float] = (
        tolerance if isinstance(tolerance, dict) else dict.fromkeys(output_names, float(tolerance))
    )
    required_names = [requirement.output for requirement in output_requirements]
    if (len(set(required_names)) != len(required_names)
            or not set(required_names) <= set(output_names)):
        raise ValueError("Output requirements must name unique declared outputs")
    if not SpecializationOptions.is_supported():
        raise RuntimeError("Reference builds require OS Core AI on macOS 27+; "
                           "set USE_OS_COREAI=1 and unset USE_LOCAL_COREAI")
    model.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("The source model must be in evaluation mode")
    inputs = {name: t.contiguous() for name, t in inputs.items()}
    arrays = runtime_inputs({name: _numpy(t) for name, t in inputs.items()})
    with torch.no_grad():
        # Later capture/replay may reuse model buffers; keep the oracle independent.
        source_outputs = tuple(t.detach().clone()
                               for t in _as_tuple(model(*tuple(inputs.values()))))
        reference = [_numpy(t) for t in source_outputs]
    if len(reference) != len(output_names):
        raise ValueError(f"model returned {len(reference)} outputs for {output_names}")
    _export_formats(
        model, inputs, output_names, source_outputs, tolerances, out_stem, profile, source_device,
        output_requirements, coreai_swaps or [], coreml_swaps or [], fp32_norm_types,
        coreml_profile,
    )
    comparisons = _compare_native(out_stem, arrays, output_names, reference, tolerances, profile,
                                  output_requirements, coreml_profile)
    write_evidence(out_stem, arrays, comparisons,
                   compute_policy=f"Core ML {coreml_profile.coreml_compute_unit}; "
                   "Core AI default specialization", source=source, build_profile=profile.value)
    write_runtime_fixture(out_stem, arrays, dict(zip(output_names, reference, strict=True)))
    print(f"wrote {out_stem.with_suffix('.aimodel')} and {out_stem.with_suffix('.mlpackage')}")
    return bool(comparisons) and all(out.verdict == Verdict.PASS for out in comparisons)
