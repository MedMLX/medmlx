"""Runtime output descriptions share a checked JSON contract."""

from pathlib import Path

import pytest

from medmlx_core.outputs import output_entry, validate_outputs


def test_output_entry_builds_a_relative_mask_description() -> None:
    entry = output_entry(
        Path("out/mask.nii.gz"), kind="label_map", coordinate_space="image_voxel",
        labels=[{"value": 0, "name": "background"}, {"value": 1, "name": "spleen"}],
        grid_of="image_path",
    )
    assert entry == {
        "path": "out/mask.nii.gz", "kind": "label_map", "coordinate_space": "image_voxel",
        "labels": [{"value": 0, "name": "background"}, {"value": 1, "name": "spleen"}],
        "grid_of": "image_path", "units": None,
    }
    assert validate_outputs([entry]) == [entry]


def test_outputs_accept_the_vocabulary_and_ignore_future_fields() -> None:
    kinds = (
        "binary_mask", "label_map", "point_set", "centerline", "detection_set", "transform",
        "correspondence_set", "feature_tensor", "measurement", "report", "preview",
        "registered_image", "synthetic_image", "class_label", "scalar_field",
    )
    spaces = (
        "image_voxel", "image_physical", "patient", "tracker", "instrument", "pixel",
        "unresolved", "not_applicable",
    )
    for kind in kinds:
        entry = output_entry("out", kind=kind, coordinate_space="unresolved", labels=[])
        assert entry["kind"] == kind
    for space in spaces:
        entry = output_entry("out", kind="report", coordinate_space=space)
        assert entry["coordinate_space"] == space
    entries = [{
        "path": "out/report.txt", "kind": "report", "coordinate_space": "not_applicable",
        "future_field": {"format": "text"},
    }]
    assert validate_outputs(entries) is entries
    assert validate_outputs([]) == []


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        pytest.param({"path": " "}, "path must be non-empty text", id="empty-path"),
        pytest.param({"kind": "mask"}, "kind must be one of", id="unknown-kind"),
        pytest.param({"coordinate_space": "voxel"}, "coordinate_space must be one of",
                     id="unknown-space"),
        pytest.param({"labels": None}, "labels is required for label_map", id="missing-labels"),
        pytest.param({"kind": "binary_mask", "labels": None}, "labels is required for binary_mask",
                     id="missing-binary-labels"),
        pytest.param({"labels": {"1": "spleen"}}, "labels must be a list", id="label-mapping"),
        pytest.param({"labels": [{"value": -1, "name": "spleen"}]},
                     "value must be a non-negative integer", id="negative-value"),
        pytest.param({"labels": [{"value": True, "name": "spleen"}]},
                     "value must be a non-negative integer", id="boolean-value"),
        pytest.param({"labels": [{"value": 1, "name": "spleen"}, {"value": 1, "name": "liver"}]},
                     "value duplicates label value 1", id="duplicate-value"),
        pytest.param({"labels": [{"value": 1, "name": " "}]}, "name must be non-empty text",
                     id="empty-name"),
        pytest.param({"grid_of": 1}, "grid_of must be non-empty text", id="invalid-grid"),
        pytest.param({"units": []}, "units must be non-empty text", id="invalid-units"),
    ],
)
def test_outputs_reject_invalid_metadata(fields: dict[str, object], reason: str) -> None:
    entry = {
        "path": "mask.nii.gz", "kind": "label_map", "coordinate_space": "image_voxel",
        "labels": [{"value": 1, "name": "spleen"}], **fields,
    }
    with pytest.raises(ValueError, match=reason):
        validate_outputs([entry])


@pytest.mark.parametrize(
    ("value", "reason"),
    [pytest.param({"mask": "mask.nii.gz"}, "outputs must be a list", id="old-mapping"),
     pytest.param(["mask.nii.gz"], r"outputs\[0\] must be an object", id="bare-path")],
)
def test_outputs_require_a_list_of_entries(value: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        validate_outputs(value)
