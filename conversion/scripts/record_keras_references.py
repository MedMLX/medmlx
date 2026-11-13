"""Regenerate real-image CPU TensorFlow oracles independently of the Torch exporter.

Run in an environment with tensorflow==2.20.0, keras==3.15.0, tf-keras==2.20.1,
numpy<2.5 and h5py==3.15.1. This script never imports an MLX model.
"""

import argparse
import ast
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeGuard, cast, runtime_checkable

import nibabel as nib
import numpy as np
import numpy.typing as npt
from PIL import Image

type FloatArray = npt.NDArray[np.float32]


@runtime_checkable
class OracleTensor(Protocol):
    """A TensorFlow eager tensor that can produce host values."""

    def numpy(self) -> object:
        """Return the evaluated values as a NumPy array."""
        ...


@runtime_checkable
class InferenceModel(Protocol):
    """A callable TF-Keras model used for one inference pass."""

    def __call__(self, inputs: FloatArray, *, training: bool) -> object:
        """Run the model on FP32 inputs with training disabled."""
        ...


@runtime_checkable
class OracleModel(InferenceModel, Protocol):
    """Consumed Keras 3.15 / TF-Keras 2.20 model methods."""

    def load_weights(self, filepath: Path) -> object:
        """Load the pinned checkpoint's weights into the model."""
        ...

    def save(self, filepath: Path, *, include_optimizer: bool) -> None:
        """Write the full model without optimizer state."""
        ...

    def to_json(self) -> str:
        """Serialize the model architecture as JSON."""
        ...


@runtime_checkable
class UNetFactory(Protocol):
    """The pinned SynthSeg ``unet`` constructor from ``ext/neuron``."""

    def __call__(
        self, *, nb_features: int, input_shape: tuple[None, None, None, int],
        nb_levels: int, conv_size: int, nb_labels: int, feat_mult: int,
        nb_conv_per_level: int, batch_norm: int, final_pred_activation: str,
    ) -> object:
        """Build the UNet from the pinned keyword arguments."""
        ...


def is_float_array(value: object) -> TypeGuard[FloatArray]:
    """Admit FP32 oracle arrays without coercing historical numerical values."""
    return isinstance(value, np.ndarray) and cast(
        npt.NDArray[np.generic], value
    ).dtype == np.dtype(np.float32)


def model_array(model: InferenceModel, image: object) -> FloatArray:
    """Validate the native eager tensor and FP32 array returned by inference."""
    if not is_float_array(image):
        raise TypeError("Oracle input must be a float32 NumPy array")
    tensor = model(image, training=False)
    if not isinstance(tensor, OracleTensor):
        raise TypeError("Oracle model must return an eager tensor with numpy()")
    result = tensor.numpy()
    if not is_float_array(result):
        raise TypeError("Oracle output must be a float32 NumPy array")
    return result


def require_model(value: object) -> OracleModel:
    """Reject non-model results at the optional Keras loading boundary."""
    if not isinstance(value, OracleModel):
        raise TypeError("Expected a callable Keras model with weight/save/JSON methods")
    return value


def load_model(path: Path) -> OracleModel:
    """Load an uncompiled Keras model using its verified public loader."""
    loader: object = getattr(getattr(import_module("keras"), "models", None), "load_model", None)
    if not callable(loader):
        raise TypeError("keras.models.load_model must be callable")
    return require_model(loader(path, compile=False))


def configure_cpu() -> None:
    """Select TensorFlow 2.20 CPU execution before constructing oracle models."""
    config = getattr(import_module("tensorflow"), "config", None)
    visible: object = getattr(config, "set_visible_devices", None)
    threads: object = getattr(
        getattr(config, "threading", None), "set_intra_op_parallelism_threads", None
    )
    if not callable(visible) or not callable(threads):
        raise TypeError("TensorFlow CPU configuration functions must be callable")
    visible([], "GPU")
    threads(4)


def provenance(checkpoint: Path, source: Path | None = None) -> dict[str, object]:
    """Identify the supplied weights and source before computing a CPU TensorFlow oracle."""
    from medmlx_convert.provenance import SourceRevision

    origin = ("https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
              if source is None else "BBillot/SynthSeg@2a2aa3bbfccb83f8253a51ca8b329b9938a2646d")
    return {
        "checkpoint": checkpoint.as_posix(),
        "origin": origin,
        "reference_source": None if source is None else source.as_posix(),
        "revision": asdict(SourceRevision.capture(Path(__file__).resolve().parents[2])),
        "oracle_packages": {name: version(name) for name in (
            "tensorflow", "keras", "tf-keras", "numpy", "h5py",
        )},
    }


def record_radimagenet(checkpoint: Path, output: Path, xray: Path) -> None:
    """Recompute RadImageNet features from a sourced chest X-ray."""
    identity = provenance(checkpoint)
    fixture = output / "radimagenet-tensorflow.npz"
    with Image.open(xray) as source:
        image = (
            np.asarray(source.convert("RGB").resize((224, 224)), dtype=np.float32)[None]
            / np.float32(255)
        )
    encoder = load_model(checkpoint)
    spatial = model_array(encoder, image)
    np.savez_compressed(
        fixture, schema_version=np.array(3), provenance=np.array(json.dumps(identity)), image=image,
        features=spatial.mean((1, 2)), spatial_features=spatial,
    )


def record_synthseg(checkpoint: Path, source: Path, output: Path, ct: Path) -> None:
    """Rebuild the pinned SynthSeg UNet, record logits, and save its graph JSON."""
    tf_keras = import_module("tf_keras")

    graph_path = output / "synthseg-v1-graph.json"
    # Identify this invocation instead of inheriting metadata from an earlier graph.
    metadata = dict(schema_version=3, provenance=provenance(checkpoint, source))
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"unet", "conv_enc", "conv_dec"}
    ]
    namespace: dict[str, object] = dict(
        np=np, keras=tf_keras, KL=getattr(tf_keras, "layers", None),
        Model=getattr(getattr(tf_keras, "models", None), "Model", None),
    )
    exec(compile(tree, str(source), "exec"), namespace)
    loader = cast(Callable[[str], object], import_module("nibabel").load)
    volume = loader(str(ct))
    if not isinstance(volume, nib.Nifti1Image):
        raise ValueError("Expected the HCC-TACE CT in NIfTI format")
    crop = np.asarray(volume.dataobj, dtype=np.float32)[100:116, 220:236, 30:46]
    if crop.shape != (16, 16, 16):
        raise ValueError(f"Unexpected CT crop shape {crop.shape}")
    image = np.clip((crop + np.float32(175)) / np.float32(425), 0, 1)[None, ..., None]
    labels_path = Path(__file__).resolve().parents[1] / "tests/fixtures/synthseg-v1-labels.json"
    labels = np.asarray(json.loads(labels_path.read_text()), dtype=np.int32)
    factory = namespace["unet"]
    if not isinstance(factory, UNetFactory):
        raise TypeError("Pinned SynthSeg unet must be callable")
    model = require_model(factory(
        nb_features=24, input_shape=(None, None, None, 1), nb_levels=5,
        conv_size=3, nb_labels=len(labels), feat_mult=2, nb_conv_per_level=2,
        batch_norm=-1, final_pred_activation="linear",
    ))
    model.load_weights(checkpoint)
    logits = model_array(model, image)
    with TemporaryDirectory() as temporary:
        full_model = Path(temporary) / "graph.h5"
        model.save(full_model, include_optimizer=False)
        metadata["model"] = json.loads(load_model(full_model).to_json())
    graph_path.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        output / "synthseg-v1-tensorflow.npz", schema_version=np.array(3),
        provenance=np.array(json.dumps(metadata["provenance"])),
        image=image, logits=logits, labels=labels,
    )


def main() -> None:
    """Record the requested oracle fixture for the given checkpoint and source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("radimagenet", "synthseg"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    cache = Path.home() / ".cache" / "medmlx" / "test-data"
    parser.add_argument("--ct", type=Path, default=cache / "hcc_063_ct.nii.gz")
    parser.add_argument("--xray", type=Path, default=cache / "chest_xray.jpg")
    args = parser.parse_args()
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args.output.mkdir(parents=True, exist_ok=True)
    configure_cpu()
    if args.model == "radimagenet":
        record_radimagenet(args.checkpoint, args.output, args.xray)
        return
    if args.source is None:
        parser.error("SynthSeg requires --source pointing to pinned ext/neuron/models.py")
    record_synthseg(args.checkpoint, args.source, args.output, args.ct)


if __name__ == "__main__":
    main()
