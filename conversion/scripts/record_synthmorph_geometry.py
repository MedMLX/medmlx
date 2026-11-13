"""Record official TensorFlow geometry outputs without importing legacy model packages.

Run in an independent TensorFlow 2.20.0 environment. Only pinned upstream function
bodies are compiled; their numerical code is not rewritten. Sourced CT pixels and
source identities are embedded in the fixture. No learned weights are involved.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from urllib.request import urlopen

import nibabel as nib
import numpy as np
import numpy.typing as npt

VXM_REVISION = "75ac2a2cd7298af3b3d563a3f0cfa000e410d099"
NEURITE_REVISION = "7547006a24b69446004a6de4e821e81a96fb56cf"


class Keras(Protocol):
    """Minimal ``tf.keras`` surface required by the compiled oracle."""

    backend: object


class TensorFlow(Protocol):
    """Minimal TensorFlow surface required by the compiled oracle."""

    keras: Keras
    constant: Callable[..., object]

    def device(self, device_name: str) -> AbstractContextManager[None]:
        """Place the enclosed graph construction on one device."""
        ...


class OracleTensor(Protocol):
    """A TensorFlow eager tensor that can produce host values."""

    def numpy(self) -> npt.NDArray[np.float32]:
        """Return the evaluated values as an FP32 host array."""
        ...


def _source(repo: Path | None, upstream: str, revision: str, path: str) -> str:
    """Read a pinned Git object locally or fetch that exact official source file."""
    if repo is None:
        url = f"https://raw.githubusercontent.com/{upstream}/{revision}/{path}"
        with urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    return subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def _functions(source: str, names: set[str], namespace: dict[str, object]) -> None:
    functions: list[ast.stmt] = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions if isinstance(node, ast.FunctionDef)} != names:
        raise ValueError("Pinned upstream oracle is missing required functions")
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<pinned-upstream>", "exec"),
         namespace)


def main() -> None:
    """Run the pinned oracle on an HCC-TACE crop and controlled pull fields.

    The .npz holds the inputs (``image`` 4x5x6 with 2 channels, ``shift`` fields
    and ``velocity``), the TensorFlow outputs ``border``, ``filled`` and
    ``integrated``, and a JSON provenance record. Runs on CPU only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxelmorph", type=Path, help="Local source Git repository")
    parser.add_argument("--neurite", type=Path, help="Local source Git repository")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ct", type=Path, default=Path(os.environ.get(
        "MEDMLX_TEST_CT", str(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"),
    )))
    options = parser.parse_args()
    if version("tensorflow") != "2.20.0":
        raise ValueError("Oracle requires TensorFlow 2.20.0")
    tf = cast(TensorFlow, import_module("tensorflow"))
    vxm = _source(options.voxelmorph, "voxelmorph/voxelmorph", VXM_REVISION,
                  "voxelmorph/utils/utils.py")
    neurite = _source(options.neurite, "adalca/neurite", NEURITE_REVISION,
                      "neurite/utils/utils.py")
    namespace: dict[str, object] = {"tf": tf, "np": np, "itertools": itertools,
                                    "K": tf.keras.backend}
    ne_names = {"interpn", "sub2ind2d", "prod_n", "volshape_to_meshgrid", "meshgrid"}
    _functions(neurite, ne_names, namespace)
    namespace["ne"] = SimpleNamespace(utils=SimpleNamespace(
        **{name: namespace[name] for name in ne_names}))
    _functions(vxm, {"transform", "integrate_vec", "is_affine_shape",
                     "validate_affine_shape"}, namespace)
    transform = namespace["transform"]
    integrate = namespace["integrate_vec"]
    if not callable(transform) or not callable(integrate):
        raise TypeError("Pinned oracle functions are not callable")
    rng = np.random.default_rng(738)
    source = nib.Nifti1Image.from_filename(options.ct)
    crop = np.asarray(source.dataobj, dtype=np.float32)[100:104, 210:215, 55:61]
    if crop.shape != (4, 5, 6):
        raise ValueError(f"Unexpected HCC-TACE crop shape {crop.shape}")
    image = np.stack((crop, crop[::-1]), axis=-1) / np.float32(1024)
    shift = rng.uniform(-1.5, 1.5, size=(3, 6, 7, 3)).astype(np.float32)
    shift[0, 0, 0] = 0
    velocity = rng.normal(0, 0.1, size=(4, 5, 6, 3)).astype(np.float32)
    constant = tf.constant
    with tf.device("/CPU:0"):
        border = transform(constant(image), constant(shift), shift_center=False)
        filled = transform(constant(image), constant(shift), fill_value=-7, shift_center=False)
        integrated = integrate(constant(velocity), method="ss", nb_steps=7)
    arrays = {"image": image, "shift": shift, "velocity": velocity}
    for name, tensor in (("border", border), ("filled", filled), ("integrated", integrated)):
        arrays[name] = cast(OracleTensor, tensor).numpy()
    metadata = {
        "schema_version": 1, "scope": "HCC-TACE dense 3D geometry; no learned model",
        "image_source": "HCC-TACE-Seg HCC_063 contrast CT / 1024 HU",
        "voxelmorph": {"revision": VXM_REVISION},
        "neurite": {"revision": NEURITE_REVISION},
        "packages": {name: version(name) for name in ("tensorflow", "keras", "numpy")},
        "seed": 738, "coordinates": "zero-based ij voxel pull fields", "steps": 7,
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(options.out, image=image, shift=shift, velocity=velocity,
             border=arrays["border"], filled=arrays["filled"], integrated=arrays["integrated"],
             schema_version=np.asarray(1, dtype=np.int32),
             provenance=np.asarray(json.dumps(metadata, sort_keys=True)))


if __name__ == "__main__":
    main()
