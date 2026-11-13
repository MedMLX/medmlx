"""Agents can turn DICOM into model input and read a mask's labels, volumes and picture."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import cast

import anyio
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from mcp import Client
from PIL import Image

import medmlx
from medmlx.jobs import Jobs
from medmlx.payloads import JobPayload, JsonValue, ModelDescription, OutputSummary
from medmlx.server import build_server
from tests._medical_images import ct_crop
from tests.conftest import add_distribution

_LABELLED = '''
LABELLED = {"id": "labelled", "label": "Labelled", "task": "segmentation",
            "run": "fakemlx:report_outputs",
            "requires_weights": False, "labels": {"1": "liver", "2": "spleen"},
            "example_arguments": {"seconds": 0}}

def report_outputs(*, seconds=0, outputs=None):
    return {} if outputs is None else {"outputs": outputs}
'''


@pytest.fixture
def labelled_site(model_site: Path) -> Path:
    add_distribution(model_site, "labelled-mlx", "[medmlx.models]\nlabelled = fakemlx:LABELLED\n")
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + _LABELLED)
    return model_site


def test_outputs_report_named_volumes_and_a_preview(labelled_site: Path, tmp_path: Path) -> None:
    mask = np.zeros((20, 20, 10), dtype=np.uint8)
    mask[2:6, 2:6, 4] = 1  # 16 voxels of 2 x 2 x 5 mm = 0.32 mL
    mask[10:12, 10:12, 4:6] = 2
    affine = np.diag([2.0, 2.0, 5.0, 1.0])
    nib.save(nib.Nifti1Image(mask, affine), tmp_path / "mask.nii.gz")
    scan = ct_crop((20, 20, 10)).astype(np.int16)
    nib.save(nib.Nifti1Image(scan, affine), tmp_path / "ct.nii.gz")

    async def body(client: Client) -> None:
        described = await client.call_tool("describe_model", {"model_id": "labelled"})
        description = cast(ModelDescription, described.structured_content)
        assert description["labels"] == {"1": "liver", "2": "spleen"}
        assert description["example_arguments"] == {"seconds": 0}
        result = await client.call_tool(
            "summarize_output", {"path": "mask.nii.gz", "model_id": "labelled"})
        summary = cast(OutputSummary, result.structured_content)
        assert summary["kind"] == "labels" and summary["spacing_mm"] == [2.0, 2.0, 5.0]
        assert summary["names_from"] == "declaration"
        assert medmlx.summarize_output("mask.nii.gz", model_id="labelled", root=tmp_path) == summary
        assert [(row["name"], row["voxels"], row["volume_ml"]) for row in summary["labels"]] == [
            ("liver", 16, 0.32), ("spleen", 8, 0.16)]
        preview = await client.call_tool(
            "preview_output", {"path": "mask.nii.gz", "image_path": "ct.nii.gz",
                               "model_id": "labelled"})
        assert preview.structured_content == summary
        image = next(item for item in preview.content if item.type == "image")
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as picture:
            assert picture.info["names_from"] == "declaration"
            assert json.loads(picture.info["labels"]) == summary["labels"]

    async def run() -> None:
        async with Client(build_server(tmp_path)) as client:
            await body(client)

    anyio.run(run)


def test_output_tools_use_run_names_for_the_resolved_file(
    labelled_site: Path, tmp_path: Path,
) -> None:
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.array([0, 1, 2], dtype=np.uint8).reshape(3, 1, 1), np.eye(4)), mask)
    alias = tmp_path / "alias.nii.gz"
    alias.symlink_to(mask)

    async def run() -> None:
        async with Client(build_server(tmp_path)) as client:
            started = await client.call_tool("run_model", {"model_id": "labelled", "arguments": {
                "outputs": [
                    {"path": "other/mask.nii.gz", "kind": "binary_mask",
                     "coordinate_space": "image_voxel", "labels": [{"value": 1, "name": "wrong"}]},
                    {"path": "./mask.nii.gz", "kind": "label_map",
                     "coordinate_space": "image_voxel",
                     "labels": [{"value": 1, "name": "prompt one"},
                                {"value": 2, "name": "prompt two"}]},
                ],
            }})
            job_id = cast(JobPayload, started.structured_content)["job_id"]
            finished = await client.call_tool("get_job", {"job_id": job_id, "wait_seconds": 5})
            job = cast(JobPayload, finished.structured_content)
            assert job["state"] == "completed", job["log_tail"]
            assert "outputs_error" not in job
            arguments = {"path": str(alias), "job_id": job_id, "model_id": "labelled"}
            result = await client.call_tool("summarize_output", arguments)
            summary = cast(OutputSummary, result.structured_content)
            assert summary["names_from"] == "outputs"
            assert summary["labels"] == [
                {"value": 1, "name": "prompt one", "voxels": 1, "volume_ml": 0.001},
                {"value": 2, "name": "prompt two", "voxels": 1, "volume_ml": 0.001},
            ]
            preview = await client.call_tool("preview_output", arguments)
            assert preview.structured_content == summary
            image = next(item for item in preview.content if item.type == "image")
            with Image.open(io.BytesIO(base64.b64decode(image.data))) as picture:
                assert picture.info["names_from"] == "outputs"
                assert json.loads(picture.info["labels"]) == summary["labels"]

    anyio.run(run)


def test_python_output_names_ignore_a_different_file_and_allow_declared_fallback(
    labelled_site: Path, tmp_path: Path,
) -> None:
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4)), path)
    arguments: dict[str, JsonValue] = {"outputs": [{
        "path": "other/mask.nii.gz", "kind": "binary_mask", "coordinate_space": "image_voxel",
        "labels": [{"value": 1, "name": "prompt name"}],
    }]}
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start("labelled", "fakemlx:report_outputs", arguments, holds_memory=False)
        job.process.wait(timeout=5)
        assert jobs.get(job.job_id).state == "completed", job.to_payload()
        unnamed = medmlx.summarize_output("mask.nii.gz", job_id=job.job_id, jobs=jobs)
        assert unnamed["names_from"] is None
        assert unnamed["labels"] == [
            {"value": 1, "name": None, "voxels": 1, "volume_ml": 0.001},
        ]
        fallback = medmlx.summarize_output(
            "mask.nii.gz", job_id=job.job_id, jobs=jobs, model_id="labelled",
        )
        assert fallback["names_from"] == "declaration"
        assert fallback["labels"][0]["name"] == "liver"
        png = medmlx.preview_output("mask.nii.gz", job_id=job.job_id, jobs=jobs)
        with Image.open(io.BytesIO(png)) as picture:
            assert picture.info["names_from"] == "null"
            assert json.loads(picture.info["labels"]) == unnamed["labels"]
    finally:
        jobs.close()


@pytest.mark.parametrize("dtype", [np.uint64, np.float64])
def test_output_summary_preserves_labels_above_signed_int64(
    tmp_path: Path, dtype: type[np.uint64] | type[np.float64],
) -> None:
    label = 2**63 + 2048
    data = np.array([0, label, label, 7], dtype=dtype).reshape(2, 2, 1)
    path = tmp_path / "large-labels.nii"
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 5.0, 1.0]), header=header), path)

    summary = medmlx.summarize_output(path, labels={label: "large label", 7: "small label"})

    assert summary["kind"] == "labels"
    assert summary["labels"] == [
        {"value": 7, "name": "small label", "voxels": 1, "volume_ml": 0.02},
        {"value": label, "name": "large label", "voxels": 2, "volume_ml": 0.04},
    ]


@pytest.mark.parametrize(("dtype", "label"), [(np.uint8, 255), (np.uint16, 65535)])
def test_output_summary_counts_sparse_unsigned_labels(
    tmp_path: Path, dtype: type[np.uint8] | type[np.uint16], label: int,
) -> None:
    data = np.array([0, 3, 3, label, label, label], dtype=dtype).reshape(3, 2, 1)
    path = tmp_path / "sparse-labels.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    summary = medmlx.summarize_output(path)

    assert summary["labels"] == [
        {"value": 3, "name": None, "voxels": 2, "volume_ml": 0.002},
        {"value": label, "name": None, "voxels": 3, "volume_ml": 0.003},
    ]


def test_output_summary_treats_many_uint16_values_as_an_image(tmp_path: Path) -> None:
    data = (ct_crop((24, 24, 24)) + 2048).astype(np.uint16)
    assert np.unique(data).size > 512
    path = tmp_path / "intensity-image.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    summary = medmlx.summarize_output(path)

    assert summary["kind"] == "image"
    assert summary["labels"] == []
    assert summary["value_range"] == [float(data.min()), float(data.max())]


@pytest.mark.parametrize("difference", ["origin", "spacing", "shear", "units"])
def test_preview_rejects_same_shape_on_different_physical_grids(
    tmp_path: Path, difference: str,
) -> None:
    mask = tmp_path / "mask.nii.gz"
    scan = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 2), np.uint8), np.eye(4)), mask)
    affine = np.eye(4)
    if difference == "origin":
        affine[0, 3] = 20
    elif difference == "spacing":
        affine[2, 2] = 2
    elif difference == "shear":
        affine[0, 1] = 0.2
    image = nib.Nifti1Image(ct_crop((3, 4, 2)).astype(np.int16), affine)
    if difference == "units":
        image.header.set_xyzt_units("meter")
    nib.save(image, scan)
    with pytest.raises(ValueError, match="affine"):
        medmlx.preview_output(mask, image_path=scan)


@pytest.mark.parametrize(("units", "scale"), [("mm", 1.0), ("meter", 0.001), ("micron", 1000.0)])
def test_output_summary_and_preview_use_physical_millimetres(
    tmp_path: Path, units: str, scale: float,
) -> None:
    mask, scan = tmp_path / "mask.nii", tmp_path / "scan.nii"
    image = nib.Nifti1Image(
        np.ones((2, 2, 2), np.uint8), np.diag([2 * scale, 3 * scale, 4 * scale, 1]),
    )
    image.header.set_xyzt_units(units)
    nib.save(image, mask)
    reference = nib.Nifti1Image(
        ct_crop((2, 2, 2)).astype(np.int16), np.diag([2.0, 3.0, 4.0, 1.0]),
    )
    reference.header.set_xyzt_units("mm")
    nib.save(reference, scan)
    summary = medmlx.summarize_output(mask)
    assert summary["spacing_mm"] == pytest.approx([2, 3, 4], rel=1e-7)
    assert summary["labels"][0]["volume_ml"] == 0.192  # 8 voxels * 24 mm^3 / 1000.
    with Image.open(io.BytesIO(medmlx.preview_output(mask, image_path=scan))) as picture:
        assert picture.format == "PNG"


def test_output_summary_uses_voxel_volume_for_sheared_affines(tmp_path: Path) -> None:
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[0, 1] = 4  # Column lengths change; each voxel still occupies 24 mm^3.
    path = tmp_path / "sheared-mask.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.uint8), affine), path)
    assert medmlx.summarize_output(path)["labels"][0]["volume_ml"] == 0.192


@pytest.fixture
def dicom_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dicom"
    folder.mkdir()
    pixels = ct_crop((4, 5, 3)).astype(np.int16)
    for index in range(3):
        plane = sitk.GetImageFromArray(pixels[:, :, index][None])
        plane.SetSpacing((0.5, 0.5, 2.0))
        plane.SetOrigin((0.0, 0.0, index * 2.0))
        for tag, value in (("0008|0060", "CT"), ("0020|000e", "1.2.3"), ("0020|000d", "1.2"),
                           ("0020|0013", str(index + 1)), ("0008|103e", "Test series")):
            plane.SetMetaData(tag, value)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()  # one series UID across the slices
        writer.SetFileName(str(folder / f"slice{index}.dcm"))
        writer.Execute(plane)
    return folder


def test_dicom_series_converts_to_nifti_once(tmp_path: Path, dicom_folder: Path) -> None:
    listed = medmlx.list_dicom_series(dicom_folder)
    assert [(row["modality"], row["files"]) for row in listed] == [("CT", 3)]
    converted = medmlx.convert_dicom(dicom_folder, tmp_path / "ct.nii.gz")
    assert converted["shape"] == [5, 4, 3] and converted["spacing_mm"] == [0.5, 0.5, 2.0]
    with pytest.raises(FileExistsError):
        medmlx.convert_dicom(dicom_folder, tmp_path / "ct.nii.gz")


@pytest.mark.parametrize("problem", ["writer-failure", "concurrent-output"])
def test_dicom_conversion_preserves_destination_on_write_failure_or_conflict(
    tmp_path: Path, dicom_folder: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
) -> None:
    target = tmp_path / "ct.nii.gz"
    original = sitk.WriteImage

    def write(image: sitk.Image, filename: str) -> None:
        if problem == "writer-failure":
            Path(filename).write_bytes(b"partial")
            raise RuntimeError("writer failed")
        target.write_bytes(b"concurrent output")
        original(image, filename)

    monkeypatch.setattr(sitk, "WriteImage", write)
    expected = RuntimeError if problem == "writer-failure" else FileExistsError
    with pytest.raises(expected):
        medmlx.convert_dicom(dicom_folder, target)
    if problem == "writer-failure":
        assert not target.exists()
    else:
        assert target.read_bytes() == b"concurrent output"
