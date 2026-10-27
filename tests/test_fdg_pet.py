"""Standalone FDG adapter contracts; Metal execution is explicit and uses tiny arrays."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from medmlx import api
from medmlx.adapters.fdg_pet import run
from medmlx.payloads import JsonValue


class FdgInputs(TypedDict):
    image_path: str
    labels_path: str
    profile_path: str
    output_dir: str


@pytest.fixture
def fdg_inputs(tmp_path: Path) -> FdgInputs:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-10, 20, 5]
    for name, data in (("ct", np.zeros((4, 5, 6))), ("labels", np.ones((4, 5, 6)))):
        image = nib.Nifti1Image(data.astype(np.float32), affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, tmp_path / f"{name}.nii.gz")
    profile: dict[str, JsonValue] = {
        "tissues": {
            "unclassified_soft_tissue": {
                "suv_mean": 0.0,
                "suv_std": 0.0,
                "maisi_labels": [],
            }
        },
        "scanner": {
            "scatter_fraction": 0.0,
            "randoms_fraction": 0.0,
            "fwhm_mm": [0.0, 0.0, 0.0],
            "counts_per_suv": 100.0,
        },
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    return {
        "image_path": str(tmp_path / "ct.nii.gz"),
        "labels_path": str(tmp_path / "labels.nii.gz"),
        "profile_path": str(tmp_path / "profile.json"),
        "output_dir": str(tmp_path / "output"),
    }


@pytest.mark.parametrize("count_level", [0.0, -1.0, float("nan"), float("inf")])
def test_fdg_rejects_invalid_count_scale_without_writing(
    fdg_inputs: FdgInputs,
    count_level: float,
) -> None:
    with pytest.raises(InvalidInputError, match="count_level"):
        run(**fdg_inputs, count_level=count_level)
    assert not Path(fdg_inputs["output_dir"]).exists()


def test_fdg_rejects_mismatched_geometry_and_invalid_profile(fdg_inputs: FdgInputs) -> None:
    labels = nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4))
    labels.header.set_xyzt_units("mm")
    nib.save(labels, fdg_inputs["labels_path"])
    with pytest.raises(InvalidInputError, match="share shape and affine"):
        run(**fdg_inputs)
    Path(fdg_inputs["profile_path"]).write_text(
        json.dumps(
            {
                "tissues": {
                    "unclassified_soft_tissue": {
                        "suv_mean": -1.0,
                        "suv_std": 0.0,
                        "maisi_labels": [],
                    }
                },
                "scanner": {},
            }
        )
    )
    with pytest.raises(InvalidInputError, match="suv_mean"):
        run(**fdg_inputs)
    assert not Path(fdg_inputs["output_dir"]).exists()


@pytest.mark.parametrize("invalid", ["units", "fractional_labels"])
def test_fdg_rejects_ambiguous_physical_inputs(fdg_inputs: FdgInputs, invalid: str) -> None:
    source = nib.load(fdg_inputs["image_path"])
    assert isinstance(source, nib.Nifti1Image)
    labels = nib.Nifti1Image(np.full((4, 5, 6), 1.5 if invalid == "fractional_labels" else 1.0),
                             source.affine)
    labels.header.set_xyzt_units("unknown" if invalid == "units" else "mm")
    nib.save(labels, fdg_inputs["labels_path"])
    with pytest.raises(InvalidInputError, match=r"units in mm|integer IDs"):
        run(**fdg_inputs)
    assert not Path(fdg_inputs["output_dir"]).exists()


@pytest.mark.skipif(os.environ.get("MEDMLX_TEST_METAL") != "1", reason="opt-in tiny Metal run")
def test_fdg_worker_preserves_geometry_and_refuses_overwrite(fdg_inputs: FdgInputs) -> None:
    arguments: dict[str, JsonValue] = {
        "image_path": fdg_inputs["image_path"], "labels_path": fdg_inputs["labels_path"],
        "profile_path": fdg_inputs["profile_path"], "output_dir": fdg_inputs["output_dir"],
    }
    job = api.run("fdg-pet-ct", arguments=arguments)
    assert job["state"] == "completed"
    destination = Path(fdg_inputs["output_dir"]) / "pet.nii.gz"
    output = nib.load(destination)
    source = nib.load(fdg_inputs["image_path"])
    assert isinstance(output, nib.Nifti1Image)
    assert isinstance(source, nib.Nifti1Image)
    assert output.affine is not None and source.affine is not None
    assert output.shape == (4, 5, 6)
    assert output.get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_array_equal(output.affine, source.affine)
    # A zero activity profile must remain zero through the real acquisition path.
    np.testing.assert_array_equal(output.get_fdata(), np.zeros((4, 5, 6)))
    original = destination.read_bytes()
    with pytest.raises(RuntimeError, match="Output exists"):
        api.run("fdg-pet-ct", arguments=arguments)
    assert destination.read_bytes() == original
    with pytest.raises(RuntimeError, match="Output must differ"):
        api.run(
            "fdg-pet-ct", arguments={**arguments, "image_path": str(destination), "overwrite": True}
        )
    assert destination.read_bytes() == original
