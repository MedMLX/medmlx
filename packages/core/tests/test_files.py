"""Output publication and argument checks shared by every runtime."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medmlx_core.errors import InvalidInputError
from medmlx_core.files import input_channels, output_targets, publish


def test_publish_restores_every_target_when_a_writer_fails(tmp_path: Path) -> None:
    mask, record = tmp_path / "mask.nii.gz", tmp_path / "record.json"
    mask.write_text("old mask")

    def fail(path: Path) -> None:
        path.write_text("partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        publish({mask: lambda path: path.write_text("new mask"), record: fail})

    assert mask.read_text() == "old mask"
    assert not record.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["mask.nii.gz"]


def test_publish_restores_replaced_targets_when_a_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask, record = tmp_path / "mask.nii.gz", tmp_path / "record.json"
    mask.write_text("old mask")
    record.write_text("old record")
    replace = os.replace

    def fail_on_record(source: Path, target: Path) -> None:
        if target == record and Path(source).parent != tmp_path:
            raise OSError("rename failed")
        replace(source, target)

    monkeypatch.setattr(os, "replace", fail_on_record)
    with pytest.raises(OSError, match="rename failed"):
        publish({
            mask: lambda path: path.write_text("new"),
            record: lambda path: path.write_text("new"),
        })

    assert (mask.read_text(), record.read_text()) == ("old mask", "old record")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mask.nii.gz", "record.json"]


def test_publish_replaces_targets_and_removes_stale_outputs(tmp_path: Path) -> None:
    mask, stale = tmp_path / "out" / "mask.nii.gz", tmp_path / "out" / "probabilities.npy"
    stale.parent.mkdir()
    stale.write_text("previous run")

    publish({mask: lambda path: path.write_text("mask")}, remove=(stale,))

    assert mask.read_text() == "mask"
    assert sorted(path.name for path in mask.parent.iterdir()) == ["mask.nii.gz"]


def test_output_targets_refuses_existing_outputs_and_input_aliases(tmp_path: Path) -> None:
    image = tmp_path / "ct.nii.gz"
    image.write_text("ct")

    with pytest.raises(InvalidInputError, match="pass overwrite=True"):
        output_targets(tmp_path, {"mask": "ct.nii.gz"}, overwrite=False)
    with pytest.raises(InvalidInputError, match="would replace input"):
        output_targets(tmp_path, {"mask": "ct.nii.gz"}, overwrite=True, inputs=[image])
    assert output_targets("~", {"mask": "m.nii.gz"}, overwrite=False) == {
        "mask": Path.home() / "m.nii.gz"
    }


def test_input_channels_requires_the_model_channels_by_name(tmp_path: Path) -> None:
    ct, pet = tmp_path / "ct.nii.gz", tmp_path / "pet.nii.gz"
    ct.write_text("ct")
    pet.write_text("pet")

    assert list(input_channels({"pet": pet, "ct": ct}, ("ct", "pet"))) == ["ct", "pet"]
    with pytest.raises(InvalidInputError, match="missing pet; unexpected suv"):
        input_channels({"ct": ct, "suv": pet}, ("ct", "pet"))
