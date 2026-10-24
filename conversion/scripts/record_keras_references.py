"""Regenerate synthetic CPU TensorFlow oracles, independently of the Torch exporter.

Run in an environment with tensorflow==2.20.0, keras==3.15.0, tf-keras==2.20.1,
numpy<2.5 and h5py==3.15.1. This script never imports an MLX model.
"""

import argparse
import ast
import json
import os
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeIs, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]


@runtime_checkable
class OracleTensor(Protocol):
    def numpy(self) -> object: ...


@runtime_checkable
class InferenceModel(Protocol):
    def __call__(self, inputs: FloatArray, *, training: bool) -> object: ...


@runtime_checkable
class OracleModel(InferenceModel, Protocol):
    """Consumed Keras 3.15 / TF-Keras 2.20 model methods."""

    def load_weights(self, filepath: Path) -> object: ...
    def save(self, filepath: Path, *, include_optimizer: bool) -> None: ...
    def to_json(self) -> str: ...


@runtime_checkable
class UNetFactory(Protocol):
    def __call__(
        self, *, nb_features: int, input_shape: tuple[None, None, None, int],
        nb_levels: int, conv_size: int, nb_labels: int, feat_mult: int,
        nb_conv_per_level: int, batch_norm: int, final_pred_activation: str,
    ) -> object: ...


def is_float_array(value: object) -> TypeIs[FloatArray]:
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
    from medmlx_convert.provenance import BuildSource

    origin = ("https://drive.google.com/file/d/1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
              if source is None else "BBillot/SynthSeg@2a2aa3bbfccb83f8253a51ca8b329b9938a2646d")
    record = asdict(BuildSource.capture(
        Path(__file__), checkpoint, origin=origin,
        configuration={} if source is None else {"reference_source": source.as_posix()},
    ))
    record["oracle_packages"] = {name: version(name) for name in (
        "tensorflow", "keras", "tf-keras", "numpy", "h5py",
    )}
    return record


def record_radimagenet(checkpoint: Path, output: Path) -> None:

    identity = provenance(checkpoint)
    fixture = output / "radimagenet-tensorflow.npz"
    with np.load(fixture, allow_pickle=False) as existing:
        image = existing["image"]
    encoder = load_model(checkpoint)
    spatial = model_array(encoder, image)
    np.savez_compressed(
        fixture, schema_version=np.array(3), provenance=np.array(json.dumps(identity)), image=image,
        features=spatial.mean((1, 2)), spatial_features=spatial,
    )


def record_synthseg(checkpoint: Path, source: Path, output: Path) -> None:
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
    with np.load(output / "synthseg-v1-tensorflow.npz", allow_pickle=False) as existing:
        image, labels = existing["image"], existing["labels"]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("radimagenet", "synthseg"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configure_cpu()
    if args.model == "radimagenet":
        record_radimagenet(args.checkpoint, args.output)
        return
    if args.source is None:
        parser.error("SynthSeg requires --source pointing to pinned ext/neuron/models.py")
    record_synthseg(args.checkpoint, args.source, args.output)


if __name__ == "__main__":
    main()
