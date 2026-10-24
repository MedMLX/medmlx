"""Export the released Keras ResNet50 through Keras's maintained Torch backend."""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    Module,
    as_array,
    as_module,
    as_tensor,
    attribute,
    function,
    library,
)
from medmlx_convert.evidence import OutputComparison, Verdict, compare_output, write_evidence
from medmlx_convert.provenance import BuildSource

CHECKPOINT = "RadImageNet-ResNet50_notop.h5"
ARCHIVE_ID = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"


def weights_path() -> Path:
    folder = Path(torch.hub.get_dir()) / "radimagenet"
    path = folder / "RadImageNet_models" / CHECKPOINT
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "models.zip"
        if not archive.is_file():
            function(library("gdown"), "download")(
                id=ARCHIVE_ID, output=str(archive), use_cookies=False
            )
        with zipfile.ZipFile(archive) as zipped:
            zipped.extract(f"RadImageNet_models/{CHECKPOINT}", folder)
    return path


class Features(Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = as_tensor(self.encoder(image, training=False))
        return spatial.mean((1, 2)), spatial


options = recipe_args("radimagenet")
out_dir = options.out
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"
keras = library("keras")
weights = weights_path()
encoder = as_module(
    function(attribute(keras, "models"), "load_model")(weights, compile=False)
)
model = Features(encoder)
fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/radimagenet-tensorflow.npz"
names = ["features", "spatial_features"]
comparisons: list[OutputComparison] = []
with np.load(fixture, allow_pickle=False) as reference:
    image = as_tensor(function(torch, "from_numpy")(as_array(reference["image"])))
    with torch.no_grad():
        outputs = model(image)
    for name, output in zip(names, outputs, strict=True):
        decision = compare_output(
            "keras_torch", name, as_array(function(output, "numpy")()),
            as_array(reference[name]), 1e-3,
        )
        comparisons.append(decision)
        print(f"Keras Torch versus TensorFlow {name}: {decision.max_abs_diff:.2e}")
    write_evidence(out_dir / "keras_bridge", {"image": as_array(reference["image"])}, comparisons,
                   reference="TensorFlow 2.20.0 / Keras 3.15.0 CPU FP32 frozen reference",
                   compute_policy="Keras Torch CPU")
if any(result.verdict != Verdict.PASS for result in comparisons):
    raise RuntimeError("Keras Torch backend differs from the independent TensorFlow reference")
passed = export_and_check(
    model, {"image": image}, names, out_dir / "radimagenet_resnet50", tolerance=1e-3,
    source=BuildSource.capture(
        Path(__file__), weights, origin=f"https://drive.google.com/file/d/{ARCHIVE_ID}",
        configuration={"tensorflow_reference": "tests/fixtures/radimagenet-tensorflow.npz"},
    ),
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
