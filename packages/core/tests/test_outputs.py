"""Runtime output descriptions share a checked JSON contract."""

import pytest

from medmlx_core.outputs import artifact, validate_outputs


def test_artifact_describes_a_mask_role_in_a_result() -> None:
    mask = artifact(
        kind="label_map", coordinate_space="image_voxel",
        labels=[{"value": 0, "name": "background"}, {"value": 1, "name": "spleen"}],
        grid_of="image_path",
    )
    assert mask == {
        "kind": "label_map", "coordinate_space": "image_voxel",
        "labels": [{"value": 0, "name": "background"}, {"value": 1, "name": "spleen"}],
        "grid_of": "image_path", "units": None,
    }
    result = {"outputs": {"mask": "out/mask.nii.gz", "record": "out/run.json"},
              "artifacts": {"mask": mask}, "details": {"voxels": 42}}
    checked = validate_outputs(result)
    assert checked.paths == {"mask": "out/mask.nii.gz", "record": "out/run.json"}
    assert checked.artifacts == {"mask": mask}
    assert validate_outputs({"outputs": {"mask": "mask.nii.gz"}}).artifacts == {}
    assert validate_outputs({}) == ({}, {})


def test_artifacts_accept_the_vocabulary_and_ignore_future_fields() -> None:
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
        assert artifact(kind=kind, coordinate_space="unresolved", labels=[])["kind"] == kind
    for space in spaces:
        entry = artifact(kind="report", coordinate_space=space)
        assert entry["coordinate_space"] == space
    future = {"kind": "report", "coordinate_space": "not_applicable",
              "future_field": {"format": "text"}}
    checked = validate_outputs({"outputs": {"report": "report.txt"},
                                "artifacts": {"report": future}})
    assert checked.artifacts["report"] is future


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
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
def test_artifacts_reject_invalid_metadata(fields: dict[str, object], reason: str) -> None:
    entry = {"kind": "label_map", "coordinate_space": "image_voxel",
             "labels": [{"value": 1, "name": "spleen"}], **fields}
    with pytest.raises(ValueError, match=reason):
        validate_outputs({"outputs": {"mask": "mask.nii.gz"}, "artifacts": {"mask": entry}})


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        pytest.param({"outputs": [{"path": "mask.nii.gz", "kind": "report",
                                   "coordinate_space": "not_applicable"}]},
                     "outputs must be an object keyed by output role", id="entry-list"),
        pytest.param({"outputs": {"mask": " "}}, "outputs.mask must be non-empty text",
                     id="empty-path"),
        pytest.param({"outputs": {"mask": ["a.nii.gz"]}}, "outputs.mask must be non-empty text",
                     id="path-list"),
        pytest.param({"outputs": {"": "mask.nii.gz"}}, "outputs role must be non-empty text",
                     id="empty-role"),
        pytest.param({"outputs": {"mask": "mask.nii.gz"}, "artifacts": []},
                     "artifacts must be an object keyed by output role", id="artifact-list"),
        pytest.param({"outputs": {"mask": "mask.nii.gz"},
                      "artifacts": {"overlay": {"kind": "preview", "coordinate_space": "pixel"}}},
                     "artifacts.overlay has no matching outputs.overlay", id="unknown-role"),
    ],
)
def test_outputs_map_roles_to_paths_that_artifacts_describe(
    result: dict[str, object], reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        validate_outputs(result)
