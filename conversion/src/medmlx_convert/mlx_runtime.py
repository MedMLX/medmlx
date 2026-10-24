"""Matched FP32 MLX adapters for the existing export fixtures and source inputs.

Models execute in their owning packages. The adapter uses the declared checkpoint
and source revision from the native build and includes
host transfer, layout adaptation and output materialization in each call.
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

import numpy as np
import torch

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
from medmlx_convert.monai_bundle import load_bundle_network
from medmlx_convert.provenance import SourceRevision
from medmlx_convert.runtime_benchmark import Call, Timing, Workload, text


class Model(StrEnum):
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


def _bundle(model: Model, origin: str, mx: ModuleType
             ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    repo, revision = origin.rsplit("@", 1)
    source = load_bundle_network(repo, revision)
    package = "flexible_unet_mlx" if model == Model.FLEXIBLE else "segresnet_mlx"
    weights_module = library(f"{package}.weights")
    network_module = library(f"{package}.network")
    if model == Model.FLEXIBLE:
        converted = as_str_dict(function(weights_module, "convert_state_dict")(
            source.model.state_dict()))
        weights = {name: function(mx, "array")(value) for name, value in converted.items()}
        native = function(network_module, "FlexibleUNet")(weights, mx=mx)
    else:
        declarations = attribute(library("segresnet_mlx.medmlx"), "BUNDLES")
        spec = attribute(function(declarations, "__getitem__")(repo.split("/")[-1]), "spec")
        converted = as_str_dict(function(weights_module, "convert_state_dict")(
            source.model.state_dict(), spec))
        weights = {name: function(mx, "array")(value) for name, value in converted.items()}
        native = function(network_module, "SegResNet")(spec, weights, mx=mx)
    function(mx, "eval")(weights)

    def infer(arrays: dict[str, Array]) -> object:
        return function(native, "__call__")(function(mx, "array")(arrays["image"]))
    return native, infer


def _liver(checkpoint: Path, mx: ModuleType
            ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    from medmlx_convert.nnunet import load_nnunet_network

    root = checkpoint.parent.parent
    source = load_nnunet_network(root)
    plans = as_str_dict(json.loads((root / "plans.json").read_text()))
    dataset = as_str_dict(json.loads((root / "dataset.json").read_text()))
    payload = as_str_dict(torch.load(checkpoint, weights_only=False,
                                    map_location="cpu"))
    configuration = text(as_str_dict(payload["init_args"])["configuration"])
    native = function(library("nnunet_mlx.model.factory"), "build_model_from_plans")(
        plans, configuration=configuration,
        input_channels=len(as_str_dict(dataset["channel_names"])),
        output_channels=len(as_str_dict(dataset["labels"])),
    )
    weights = library("nnunet_mlx.model.weights")
    converted = function(weights, "convert_state_dict")(source.model.state_dict())
    function(weights, "load_converted_state")(native, converted)
    function(native, "eval")()
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
    function(native, "load_h5")(checkpoint)
    function(native, "eval")()
    function(mx, "eval")(function(native, "parameters")())

    def infer(arrays: dict[str, Array]) -> object:
        logits = function(native, "logits")(function(mx, "array")(arrays["image"]))
        return function(mx, "transpose")(logits, (0, 4, 1, 2, 3))
    return native, infer


def _vista(workload: Workload, checkpoint: Path, mx: ModuleType
            ) -> tuple[object, Callable[[dict[str, Array]], object]]:
    weights = function(library("mlx_segment_ct.mlx_weights"),
                       "convert_weights_from_safetensors")(checkpoint, mx=mx)
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


def load_mlx(workload: Workload, root: Path) -> MLXSession:
    """Load the actual port with the native build's checkpoint and CPU FP32 oracle."""
    if workload.build_profile != BuildProfile.FP32:
        raise ValueError("These MLX ports are FP32; an FP16 comparison is not precision-matched")
    model = Model(workload.stem.name)
    sources = _source_revisions(root, model)
    mx = library("mlx.core")
    if function(attribute(mx, "metal"), "is_available")() is not True:
        raise RuntimeError("MLX Metal is required; no CPU fallback")
    function(mx, "set_default_device")(attribute(mx, "gpu"))
    checkpoint, origin = _checkpoint(workload)
    # Record the active adapter revision separately from the earlier export.
    sources = (*sources, SourceRevision.capture(Path(__file__).resolve().parents[2]))
    if model in (Model.FLEXIBLE, Model.RENAL):
        native, invoke = _bundle(model, origin, mx)
    elif model == Model.LIVER:
        native, invoke = _liver(checkpoint, mx)
    elif model == Model.SYNTHSEG:
        native, invoke = _synthseg(workload, checkpoint, mx)
    else:
        native, invoke = _vista(workload, checkpoint, mx)
    function(mx, "synchronize")()
    return MLXSession(native, workload, invoke, mx, sources)
