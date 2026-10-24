"""Export SynthSeg 1.0 logits; Keras Torch is checked against original TF-Keras."""

import json
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    Module,
    as_array,
    as_ints,
    as_module,
    as_tensor,
    attribute,
    class_,
    function,
    item,
    library,
)
from medmlx_convert.evidence import Verdict, compare_output, write_evidence
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import SeparableMaxPool3d

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"
keras = library("keras")
layers = attribute(keras, "layers")
Pooling = cast(type[Module], class_(layers, "MaxPooling3D"))
Upsampling = cast(type[Module], class_(layers, "UpSampling3D"))


class ExportPool(Pooling):
    """The released graph's even-sized, non-overlapping 2x2x2 max pooling."""

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        if (as_ints(attribute(self, "pool_size")) != (2, 2, 2)
                or as_ints(attribute(self, "strides")) != (2, 2, 2)
                or attribute(self, "data_format") != "channels_last"
                or any(size % 2 for size in inputs.shape[1:4])):
            raise ValueError("Unexpected SynthSeg pooling contract")
        pool = SeparableMaxPool3d(torch.nn.MaxPool3d(2))
        return as_tensor(pool(inputs.permute(0, 4, 1, 2, 3))).permute(0, 2, 3, 4, 1)


class ExportUpsample(Upsampling):
    """Avoid Keras repeat-interleave's rank-six intermediate in Core ML."""

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        if attribute(self, "data_format") != "channels_last":
            raise ValueError("Unexpected SynthSeg upsampling layout")
        output = function(torch.nn.functional, "interpolate")(
            inputs.permute(0, 4, 1, 2, 3),
            scale_factor=as_ints(attribute(self, "size")), mode="nearest",
        )
        return as_tensor(output).permute(0, 2, 3, 4, 1)


def clone_layer(layer: object) -> object:
    for original, replacement in ((Pooling, ExportPool), (Upsampling, ExportUpsample)):
        if isinstance(layer, original):
            return function(replacement, "from_config")(function(layer, "get_config")())
    return layer


class Segmentation(Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return as_tensor(self.encoder(image, training=False)).permute(0, 4, 1, 2, 3)


options = recipe_args("synthseg")
out_dir = options.out
register = function(attribute(keras, "saving"), "register_keras_serializable")(
    package="medmlx_convert"
)
for layer_type in (ExportPool, ExportUpsample):
    function(register, "__call__")(layer_type)
fixtures = Path(__file__).resolve().parents[2] / "tests/fixtures"
graph = json.loads((fixtures / "synthseg-v1-graph.json").read_text())
if item(graph, "schema_version") != 3:
    raise ValueError("Unsupported SynthSeg graph schema")
encoder = function(attribute(keras, "models"), "model_from_json")(json.dumps(item(graph, "model")))
weights = Path(torch.hub.get_dir()) / "synthseg_1.0.h5"
if not weights.is_file():
    torch.hub.download_url_to_file(
        "https://raw.githubusercontent.com/BBillot/SynthSeg/"
        "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/models/synthseg_1.0.h5",
        str(weights),
    )
function(encoder, "load_weights")(weights)
encoder = function(attribute(keras, "models"), "clone_model")(encoder, clone_function=clone_layer)
model = Segmentation(as_module(encoder)).eval()
with np.load(fixtures / "synthseg-v1-tensorflow.npz", allow_pickle=False) as reference:
    image = as_tensor(function(torch, "from_numpy")(as_array(reference["image"])))
    with torch.no_grad():
        actual = as_array(function(as_tensor(model(image)), "numpy")())
    decision = compare_output(
        "keras_torch", "logits", actual,
        as_array(reference["logits"]).transpose(0, 4, 1, 2, 3), 1e-3,
    )
    write_evidence(
        out_dir / "keras_bridge", {"image": as_array(reference["image"])}, [decision],
        reference="TensorFlow 2.20.0 / TF-Keras 2.20.1 CPU FP32 frozen reference",
        compute_policy="Keras Torch CPU with exact pooling and upsampling rewrites",
    )
    print(f"Keras Torch versus TensorFlow logits: {decision.max_abs_diff:.2e}")
    if decision.verdict != Verdict.PASS:
        raise RuntimeError("SynthSeg differs from the original TensorFlow graph")
image = torch.rand((1, 64, 64, 64, 1), generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    model, {"image": image}, ["logits"], out_dir / "synthseg_v1", tolerance=1e-3,
    source=BuildSource.capture(
        Path(__file__), weights,
        origin="BBillot/SynthSeg@2a2aa3bbfccb83f8253a51ca8b329b9938a2646d",
        configuration={"graph": "tests/fixtures/synthseg-v1-graph.json",
                       "tensorflow_reference": "tests/fixtures/synthseg-v1-tensorflow.npz"},
    ),
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
