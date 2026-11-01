"""Matched FP32 MLX adapters for the existing export fixtures and source inputs.

Models execute in their owning packages. Native safetensors weights are supplied
explicitly for converted-only ports and checked against the independent source
outputs. Each call includes host transfer, layout adaptation and materialization.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import cast

import numpy as np

from medmlx_convert.boundary import (
    Array,
    as_array,
    as_ints,
    as_str_dict,
    attribute,
    function,
    library,
)
from medmlx_convert.guidance import BuildProfile
from medmlx_convert.provenance import SourceRevision
from medmlx_convert.runtime_benchmark import Call, Timing, Workload, text


class Model(StrEnum):
    """Native MLX port selected by its export fixture name."""

    FLEXIBLE = "flexible_unet_endoscopic"
    RENAL = "renalStructures_CECT_segmentation"
    LIVER = "nnunet_liver"
    SYNTHSEG = "synthseg_v1"
    VISTA = "vista3d_auto"


_PORTS = {
    Model.FLEXIBLE: "flexible-unet-mlx", Model.RENAL: "segresnet-mlx",
    Model.LIVER: "nnunet-mlx",
    Model.SYNTHSEG: "synthseg-mlx", Model.VISTA: "mlx-segment-ct",
}


def _source_revisions(root: Path, model: Model) -> tuple[SourceRevision, ...]:
    """Record explicit checkout revisions before importing their model code."""
    if root.is_absolute():
        raise ValueError("Use a repository-relative MLX checkout root")
    sources: list[SourceRevision] = []
    for repo in ("medmlx-core", _PORTS[model]):
        directory = root / repo
        if not (directory / "src").is_dir():
            raise FileNotFoundError(f"Missing MLX source checkout: {directory}")
        sources.append(SourceRevision.capture(directory))
        sys.path.insert(0, str((directory / "src").resolve()))
    return tuple(sources)


def _checkpoint(workload: Workload) -> tuple[Path, str]:
    """Read the checkpoint path and release recorded by the native source loader."""
    raw = as_str_dict(json.loads(workload.stem.with_suffix(".verification.json").read_text()))
    source = as_str_dict(raw["source"])
    return Path(text(source["checkpoint"])), text(source["origin"])


def _bundle(model: Model, origin: str, mx: ModuleType, weights_path: Path | None
            ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    repo, _ = origin.rsplit("@", 1)
    if weights_path is None:
        raise ValueError(f"{model.value} comparison requires --mlx-weights")
    if model == Model.FLEXIBLE:
        native = function(library("flexible_unet_mlx.weights"), "load_model")(weights_path)
    else:
        declarations = attribute(library("segresnet_mlx.medmlx"), "BUNDLES")
        spec = attribute(function(declarations, "__getitem__")(repo.split("/")[-1]), "spec")
        native = function(library("segresnet_mlx.weights"), "load_model")(weights_path, spec=spec)

    def infer(arrays: dict[str, Array]) -> object:
        return function(native, "__call__")(function(mx, "array")(arrays["image"]))
    return native, infer


def _liver(checkpoint: Path, mx: ModuleType, weights_path: Path
            ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    from medmlx_convert.nnunet import load_nnunet_network

    root = checkpoint.parent.parent
    source = load_nnunet_network(root)
    plans = as_str_dict(json.loads((root / "plans.json").read_text()))
    dataset = as_str_dict(json.loads((root / "dataset.json").read_text()))
    native = function(library("nnunet_mlx.model.factory"), "build_model_from_plans")(
        plans, configuration=source.configuration,
        input_channels=len(as_str_dict(dataset["channel_names"])),
        output_channels=len(as_str_dict(dataset["labels"])),
    )
    function(native, "load_weights")(str(weights_path), strict=True)
    function(native, "eval")()
    function(mx, "eval")(function(native, "parameters")())
    runtime = library("nnunet_mlx.runtime")
    compilation = function(runtime, "default_inference_compilation")(native)
    if not isinstance(compilation, bool):
        raise TypeError("nnU-Net compilation policy must return a boolean")
    if compilation:
        # Match ModelFolderPredictor's qualified native-arithmetic default.
        native = function(runtime, "compile_inference_model")(native)

    def infer(arrays: dict[str, Array]) -> object:
        value = function(mx, "transpose")(function(mx, "array")(arrays["image"]),
                                           (0, 2, 3, 4, 1))
        return function(mx, "transpose")(function(native, "__call__")(value), (0, 4, 1, 2, 3))
    return native, infer


def _synthseg(workload: Workload, checkpoint: Path, mx: ModuleType
               ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    native = function(library("synthseg_mlx.mlx_model"), "UNet")(
        1, workload.references["logits"].shape[1], "unet")
    function(native, "load_weights")(str(checkpoint), strict=True)
    function(native, "eval")()
    function(mx, "eval")(function(native, "parameters")())

    def infer(arrays: dict[str, Array]) -> object:
        logits = function(native, "logits")(function(mx, "array")(arrays["image"]))
        return function(mx, "transpose")(logits, (0, 4, 1, 2, 3))
    return native, infer


def _vista(workload: Workload, checkpoint: Path, mx: ModuleType
            ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    shapes = function(library("mlx_segment_ct.mlx_network"), "required_shapes")()
    weights = function(library("mlx_segment_ct.medmlx"), "load_weights")(mx, checkpoint, shapes)
    classes = as_ints(workload.arrays["class_vector"].reshape(-1).tolist())
    native = function(library("mlx_segment_ct.mlx_network"), "AutomaticModel")(
        weights, classes, mx=mx)
    function(mx, "eval")(weights)

    def infer(arrays: dict[str, Array]) -> object:
        # Supplying IDs includes the learned prompt path on every invocation.
        classes = as_ints(arrays["class_vector"].reshape(-1).tolist())
        return function(native, "__call__")(function(mx, "array")(arrays["image"]), classes)
    return native, infer


@dataclass(frozen=True)
class MLXSession:
    """A loaded native port, its workload, and the captured source revisions."""

    model: object
    workload: Workload
    invoke: Callable[[dict[str, Array]], object]
    mx: ModuleType
    implementation: tuple[SourceRevision, ...]

    async def infer(self) -> Call:
        """Measure host input to evaluated, synchronized, completed host output."""
        function(self.mx, "synchronize")()
        started = perf_counter()
        output = self.invoke(self.workload.arrays)
        if not isinstance(output, np.ndarray):
            function(self.mx, "eval")(output)
        executed = perf_counter()
        host = as_array(function(np, "asarray")(output))
        function(self.mx, "synchronize")()
        finished = perf_counter()
        return Call(Timing(0, executed - started, finished - executed, finished - started),
                    {"logits": host})


def load_mlx(workload: Workload, root: Path, weights_path: Path | None = None) -> MLXSession:
    """Load the port with explicit native weights and the independent CPU FP32 oracle."""
    if workload.build_profile != BuildProfile.FP32:
        raise ValueError("These MLX ports are FP32; an FP16 comparison is not precision-matched")
    model = Model(workload.stem.name)
    if model in (Model.FLEXIBLE, Model.RENAL, Model.LIVER, Model.SYNTHSEG, Model.VISTA):
        if weights_path is None or weights_path.suffix != ".safetensors":
            raise ValueError(f"{model.value} requires a native --mlx-weights safetensors file")
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
    sources = _source_revisions(root, model)
    mx = library("mlx.core")
    if function(attribute(mx, "metal"), "is_available")() is not True:
        raise RuntimeError("MLX Metal is required; no CPU fallback")
    function(mx, "set_default_device")(attribute(mx, "gpu"))
    checkpoint, origin = _checkpoint(workload)
    # Record the active adapter revision separately from the earlier export.
    sources = (*sources, SourceRevision.capture(Path(__file__).resolve().parents[2]))
    if model in (Model.FLEXIBLE, Model.RENAL):
        native, invoke = _bundle(model, origin, mx, weights_path)
    elif model == Model.LIVER:
        native, invoke = _liver(checkpoint, mx, cast(Path, weights_path))
    elif model == Model.SYNTHSEG:
        native, invoke = _synthseg(workload, cast(Path, weights_path), mx)
    else:
        native, invoke = _vista(workload, cast(Path, weights_path), mx)
    function(mx, "synchronize")()
    return MLXSession(native, workload, invoke, mx, sources)
