"""Spatial grounding, evidence persistence and tool authority at the workflow boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core import weights
from numpy.typing import NDArray

from medmlx import registry
from medmlx.payloads import JsonValue
from medmlx.workflows.ct_agent import parse_action, tool_choice
from medmlx.workflows.ct_contracts import Candidate
from medmlx.workflows.ct_geometry import (
    Scan,
    display_plane,
    ground_candidates,
    prepare_scan,
    read_nifti,
    restore_mask,
)
from medmlx.workflows.reason_ct import run
from tests._medical_images import ct_crop


def image_fixture(directory: Path, orientation: str = "ras") -> tuple[Path, NDArray[np.uint8]]:
    """Place a real CT crop and a controlled ROI on the requested physical grid."""
    labels = np.zeros((11, 13, 9), dtype=np.uint8)
    labels[2:5, 7:9, 3:6] = 1
    data = ct_crop((11, 13, 9))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-50, -80, -100]
    units = "mm"
    if orientation == "lps":
        labels = labels[::-1, ::-1, :]
        data = data[::-1, ::-1, :]
        affine = affine @ np.asarray(
            [[-1, 0, 0, 10], [0, -1, 0, 12], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
    elif orientation == "permuted":
        labels = labels.transpose(2, 0, 1)
        data = data.transpose(2, 0, 1)
        affine = affine @ np.asarray(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )
    elif orientation in {"meter", "micron"}:
        units = orientation
        affine[:3] *= 0.001 if units == "meter" else 1000
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units(units)
    path = directory / "ct.nii.gz"
    nib.save(image, path)
    return path, labels


def detection_file(directory: Path, source: Path, *, box: list[float] | None = None) -> Path:
    record = {
        "schema_version": "medmlx.detections.v1",
        "model_id": "lung_nodule_ct_detection",
        "source_format": "monai.lung_nodule_ct_detection.v0.6.10",
        "coordinate_space": "image_physical",
        "coordinate_convention": "nifti_ras",
        "source_coordinate_space": "dicom_patient_lps",
        "units": "mm",
        "box_mode": "cccwhd",
        "classes": {"0": "nodule"},
        "source_output": "bundle_detections.json",
        "detections": [
            {
                "box": box or [-44.0, -57.5, -84.0, 6.0, 6.0, 12.0],
                "label": 0,
                "score": 0.9,
                "source_image": str(source.resolve()),
            }
        ],
    }
    path = directory / "detections.json"
    path.write_text(json.dumps(record))
    return path


def ground(scan: Scan, path: Path, padding_mm: float = 0) -> tuple[list[Candidate], int]:
    return ground_candidates(scan, path, threshold=0.25, limit=5, padding_mm=padding_mm)


@pytest.mark.parametrize("orientation", ["lps", "permuted", "micron"])
def test_detector_prompt_and_mask_roundtrip_in_physical_space(
    tmp_path: Path, orientation: str,
) -> None:
    source, expected = image_fixture(tmp_path, orientation)
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    candidates, outside = ground(scan, detection_file(tmp_path, scan.working_path))
    candidate = candidates[0]
    np.testing.assert_allclose(candidate.box_xyxy, [1.5, 6.5, 4.5, 8.5], atol=2e-6)
    assert candidate.frame_idx == 4 and outside == 0 and not scan.resampled
    # Reorient the controlled labels independently of the workflow's image handling.
    labels = np.asarray(nib.as_closest_canonical(
        nib.Nifti1Image(expected, scan.original.affine),
    ).get_fdata(), dtype=np.uint8)
    mask_path = tmp_path / "working_mask.nii.gz"
    nib.save(nib.Nifti1Image(labels, scan.working.affine, scan.working.header), mask_path)
    measurement = restore_mask(scan, candidate, mask_path, tmp_path / "native.nii.gz")
    restored = read_nifti(tmp_path / "native.nii.gz")
    np.testing.assert_array_equal(restored.get_fdata(), expected)
    np.testing.assert_array_equal(restored.affine, scan.original.affine)
    assert restored.header.get_xyzt_units() == scan.original.header.get_xyzt_units()
    assert measurement.voxel_count == 18
    assert measurement.volume_mm3 == pytest.approx(432.0, rel=1e-6)
    np.testing.assert_allclose(measurement.centroid_ras_mm, [-44, -57.5, -84], atol=3e-6)
    plane = display_plane(labels, 2, 4)
    assert plane[13 - 1 - 7, 11 - 1 - 2] == 1
    assert plane[7, 2] == 0  # An unflipped overlay would select the opposite side.


def test_candidate_identity_survives_sorting_and_filtering(tmp_path: Path) -> None:
    source, _ = image_fixture(tmp_path)
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    path = detection_file(tmp_path, scan.working_path)
    document: dict[str, JsonValue] = json.loads(path.read_text())
    row = cast(list[dict[str, JsonValue]], document["detections"])[0]
    document["detections"] = [
        {**row, "score": 0.3},
        {**row, "score": 0.9, "box": [-36.0, -57.5, -84.0, 6.0, 6.0, 12.0]},
        {**row, "score": 0.95, "box": [1000.0, 1000.0, 1000.0, 6.0, 6.0, 12.0]},
        {**row, "score": 0.01},
    ]
    path.write_text(json.dumps(document))
    candidates, outside = ground(scan, path)
    assert [c.candidate_id for c in candidates] == ["candidate-002", "candidate-001"]
    assert outside == 1
    np.testing.assert_allclose(candidates[0].box_xyxy, [5.5, 6.5, 8.5, 8.5])


@pytest.mark.parametrize("fault", ["source", "units", "classes"])
def test_detection_provenance_and_coordinate_contract_are_required(
    tmp_path: Path, fault: str,
) -> None:
    source, _ = image_fixture(tmp_path)
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    path = detection_file(tmp_path, scan.working_path)
    document: dict[str, JsonValue] = json.loads(path.read_text())
    if fault == "source":
        cast(list[dict[str, JsonValue]], document["detections"])[0]["source_image"] = str(source)
    elif fault == "units":
        document["units"] = "voxel"
    else:
        document["classes"] = {"0": "other"}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        ground(scan, path)


def test_oblique_resampling_preserves_a_world_space_sphere(tmp_path: Path) -> None:
    angle = np.pi / 6
    affine = np.asarray(
        [
            [2 * np.cos(angle), -3 * np.sin(angle), 0, -30],
            [2 * np.sin(angle), 3 * np.cos(angle), 0, -40],
            [0, 0, 4, -50],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    shape = (21, 23, 19)
    original = nib.Nifti1Image(ct_crop(shape), affine)
    original.header.set_xyzt_units("mm")
    source = tmp_path / "ct.nii.gz"
    nib.save(original, source)
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    assert scan.resampled and scan.working.affine is not None
    centre = affine[:3, :3] @ np.asarray([10, 11, 9]) + affine[:3, 3]
    radius = 10.0
    grid = np.indices(scan.working.shape).reshape(3, -1)
    world = scan.working.affine[:3, :3] @ grid + scan.working.affine[:3, 3, None]
    mask = (np.linalg.norm(world - centre[:, None], axis=0) <= radius).reshape(scan.working.shape)
    box = [*map(float, centre), 20.0, 20.0, 20.0]
    candidates, _ = ground(scan, detection_file(tmp_path, scan.working_path, box=box), 4)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), scan.working.affine, scan.working.header),
             mask_path)
    measurement = restore_mask(scan, candidates[0], mask_path, tmp_path / "native.nii.gz")
    native = read_nifti(tmp_path / "native.nii.gz")
    grid = np.indices(shape).reshape(3, -1)
    distance = np.linalg.norm(affine[:3, :3] @ grid + affine[:3, 3, None] - centre[:, None], axis=0)
    # Nearest-neighbour sampling may differ only within half a working voxel diagonal.
    boundary = np.linalg.norm([2.0, 3.0, 4.0]) / 2
    values = native.get_fdata().ravel()
    assert np.all(values[distance < radius - boundary] == 1)
    assert np.all(values[distance > radius + boundary] == 0)
    assert np.linalg.norm(np.asarray(measurement.centroid_ras_mm) - centre) < boundary
    np.testing.assert_array_equal(native.affine, read_nifti(source).affine)


def test_unknown_units_require_an_explicit_assumption(tmp_path: Path) -> None:
    source, _ = image_fixture(tmp_path)
    image = read_nifti(source)
    image.header.set_xyzt_units("unknown")
    nib.save(image, source)
    with pytest.raises(ValueError):
        prepare_scan(source, tmp_path / "working.nii.gz")
    scan = prepare_scan(source, tmp_path / "working.nii.gz", assume_mm=True)
    assert scan.assumed_mm and scan.working.header.get_xyzt_units()[0] == "mm"


def test_axis_aligned_mask_edge_voxels_are_never_resampled(tmp_path: Path) -> None:
    source, _ = image_fixture(tmp_path, "meter")
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    box = [-40.0, -62.0, -84.0, 100.0, 100.0, 100.0]
    candidates, _ = ground(scan, detection_file(tmp_path, scan.working_path, box=box))
    labels = np.ones(scan.working.shape, dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(labels, scan.working.affine, scan.working.header), mask_path)
    measurement = restore_mask(scan, candidates[0], mask_path, tmp_path / "native.nii.gz")
    assert measurement.voxel_count == 11 * 13 * 9
    np.testing.assert_array_equal(read_nifti(tmp_path / "native.nii.gz").get_fdata(), labels)


@pytest.mark.parametrize("fault", ["wrong_grid", "wrong_label", "outside_prompt"])
def test_invalid_segmentations_cannot_be_measured(tmp_path: Path, fault: str) -> None:
    source, labels = image_fixture(tmp_path)
    scan = prepare_scan(source, tmp_path / "working.nii.gz")
    candidates, _ = ground(scan, detection_file(tmp_path, scan.working_path))
    assert scan.working.affine is not None
    affine = scan.working.affine.copy()
    if fault == "wrong_grid":
        affine[0, 3] += 1
    elif fault == "wrong_label":
        labels *= 2
    else:
        labels[:] = 0
        labels[8, 2, 4] = 1
    mask_path = tmp_path / "bad_mask.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine, scan.working.header), mask_path)
    with pytest.raises(ValueError):
        restore_mask(scan, candidates[0], mask_path, tmp_path / "native.nii.gz")
    assert not (tmp_path / "native.nii.gz").exists()


def providers(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    """Replace external inference only; geometry, dispatch policy and artifacts stay real."""

    def reason(*, output_dir: Path, response_choices: object = None, **arguments: object) -> object:
        output_dir.mkdir(parents=True)
        response = (
            responses.pop(0)
            if response_choices is not None
            else "Candidate-001 was segmented. <script>alert('unsafe')</script>"
        )
        (output_dir / "report.txt").write_text(response)
        return {}

    def detect(
        *, image_path: Path, weights_path: Path, output_dir: Path, resample_raw_input: bool
    ) -> object:
        """Exercise the current detector's explicit runner arguments."""
        assert resample_raw_input
        output_dir.mkdir(parents=True)
        detection_file(output_dir, image_path)
        return {}

    def segment(*, image_path: Path, output_dir: Path, **arguments: object) -> object:
        image = read_nifti(image_path)
        labels = np.zeros(image.shape, dtype=np.uint8)
        labels[2:5, 7:9, 3:6] = 1
        output_dir.mkdir(parents=True, exist_ok=False)
        nib.save(
            nib.Nifti1Image(labels, image.affine, image.header), output_dir / "medsam2_mask.nii.gz"
        )
        return {}

    def spec(name: str) -> registry.ModelSpec:
        return registry.ModelSpec(name, name, "test", name, None, None, None, None, name)

    implementations: dict[str, Callable[..., object]] = {
        "mlx-reason-ct": reason,
        "lung-nodule": detect,
        "medsam2": segment,
    }
    monkeypatch.setattr(registry, "get_model", spec)
    monkeypatch.setattr(registry, "resolve", implementations.__getitem__)


def run_fixture(tmp_path: Path, *, max_steps: int = 8) -> dict[str, object]:
    source, _ = image_fixture(tmp_path, "lps")
    weight = tmp_path / "weights.safetensors"
    weight.touch()
    return run(
        image_path=source,
        reason_weights_path=tmp_path,
        detector_weights_path=weight,
        medsam_weights_path=weight,
        output_dir=tmp_path / "out",
        max_steps=max_steps,
        padding_mm=0.0,
    )


def test_workflow_rejects_bad_actions_and_persists_grounded_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers(
        monkeypatch,
        [
            tool_choice("detect_nodules"),
            tool_choice("segment_candidate", "../../escape"),
            tool_choice("segment_candidate", "candidate-001"),
            tool_choice("segment_candidate", "candidate-001"),
            tool_choice("finish"),
        ],
    )
    result = run_fixture(tmp_path)
    details = cast(dict[str, object], result["details"])
    assert details["status"] == "completed"
    saved: dict[str, object] = json.loads((tmp_path / "out/run.json").read_text())
    assert saved == details
    measurements = cast(list[dict[str, object]], saved["measurements"])
    assert measurements[0]["volume_mm3"] == pytest.approx(432.0)
    assert measurements[0]["centroid_ras_mm"] == [-44.0, -57.5, -84.0]
    events = cast(list[dict[str, object]], saved["events"])
    assert sum(event["event"] == "request_rejected" for event in events) == 2
    page = (tmp_path / "out/review.html").read_text()
    assert "&lt;script&gt;alert" in page and "<script>alert" not in page
    assert (tmp_path / "out/candidate-001/mask.nii.gz").is_file()
    assert not (tmp_path / "escape").exists()


def test_omitted_weights_use_each_models_automatic_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers(
        monkeypatch,
        [
            tool_choice("detect_nodules"),
            tool_choice("segment_candidate", "candidate-001"),
            tool_choice("finish"),
        ],
    )
    folders: dict[str, Path] = {}

    def download(model_id: str) -> Path:
        folders[model_id] = tmp_path / "cache" / model_id
        folders[model_id].mkdir(parents=True)
        return folders[model_id]

    monkeypatch.setattr(weights, "weights_dir", download)
    source, _ = image_fixture(tmp_path, "lps")
    run(image_path=source, output_dir=tmp_path / "out", padding_mm=0.0)
    saved: dict[str, object] = json.loads((tmp_path / "out/run.json").read_text())
    assert list(folders) == ["mlx-reason-ct", "lung-nodule", "medsam2"]
    assert saved["weights"] == {name: str(path.resolve()) for name, path in folders.items()}


def test_explicit_weights_never_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    providers(
        monkeypatch,
        [
            tool_choice("detect_nodules"),
            tool_choice("segment_candidate", "candidate-001"),
            tool_choice("finish"),
        ],
    )

    def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("Explicit weight paths must be used as given")

    monkeypatch.setattr(weights, "weights_dir", unexpected)
    run_fixture(tmp_path)


def test_exhausted_decisions_leave_an_incomplete_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers(monkeypatch, [tool_choice("finish")] * 3)
    with pytest.raises(RuntimeError):
        run_fixture(tmp_path, max_steps=3)
    record = json.loads((tmp_path / "out/run.json").read_text())
    assert record["status"] == "incomplete"
    assert not (tmp_path / "out/review.html").exists()


def test_workflow_dispatches_to_released_lung_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workflow reaches the detector's asset check with its real runner API."""
    from lung_nodule_mlx.medmlx import run as detector_run
    from medmlx_core.errors import AssetNotReadyError

    providers(monkeypatch, [tool_choice("detect_nodules")])
    resolve = registry.resolve

    def resolve_with_detector(name: str) -> Callable[..., object]:
        return detector_run if name == "lung-nodule" else resolve(name)

    monkeypatch.setattr(registry, "resolve", resolve_with_detector)
    with pytest.raises(AssetNotReadyError):
        run_fixture(tmp_path)
    record = json.loads((tmp_path / "out/run.json").read_text())
    assert record["status"] == "incomplete"


@pytest.mark.parametrize(
    "text",
    [
        "Detect nodules",
        tool_choice("shell"),
        tool_choice("segment_candidate", "candidate-001") + "suffix",
    ],
)
def test_only_complete_allowed_tool_calls_are_admitted(text: str) -> None:
    with pytest.raises(ValueError):
        parse_action(text)
