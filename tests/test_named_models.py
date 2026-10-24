"""Named checkpoint admission and public native-provider dispatch contracts."""

import json
from pathlib import Path

import pytest
from medmlx_core.errors import AssetNotReadyError

from medmlx.adapters import nnunet


@pytest.mark.parametrize("model", ["pancreas", "brats", "mrsegmentator"])
@pytest.mark.parametrize("wrong_identity", [False, True])
def test_named_checkpoint_admission_and_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, wrong_identity: bool,
) -> None:
    datasets = {
        "pancreas": ("Dataset999_PancCTPretrain", {"0": "CT"},
                     {"background": 0, "pancreas": 1, "cancer": 2}),
        "brats": ("Dataset002_BRATS19", {"0": "t1", "1": "t1ce", "2": "t2", "3": "flair"},
                  {"background": 0, "edema": 1, "nonenhancing": 2, "empty": 3, "enhancing": 4}),
        "mrsegmentator": ("Dataset050_ukbb_inhouse_nako", {"0": "MR"},
                          {"background": 0, "spleen": 1}),
    }
    dataset, channels, labels = datasets[model]
    (tmp_path / "plans.json").write_text(json.dumps({
        "dataset_name": "DatasetOther" if wrong_identity else dataset,
    }))
    (tmp_path / "dataset.json").write_text(json.dumps({
        "channel_names": channels, "labels": labels,
    }))
    calls: list[dict[str, object]] = []

    def predict(**arguments: object) -> dict[str, object]:
        calls.append(arguments)
        return {"outputs": {"segmentation": "out/mask.nii.gz"}, "details": {"labels": labels}}

    monkeypatch.setattr("nnunet_mlx.medmlx.run", predict)
    arguments = {"model_dir": str(tmp_path), "output_dir": "out"}
    if model == "brats":
        arguments.update({
            f"{role}_path": f"{role}.nii.gz" for role in ("flair", "t1", "t1ce", "t2")
        })
    else:
        arguments["image_path"] = "image.nii.gz"
    runner = getattr(nnunet, model)
    if wrong_identity:
        with pytest.raises(AssetNotReadyError, match="must match"):
            runner(**arguments)
        assert calls == []
        return
    result = runner(**arguments)
    assert result["outputs"] == {"mask": "out/mask.nii.gz"}
    assert calls[0]["image_paths"] == (
        ["t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz", "flair.nii.gz"]
        if model == "brats" else ["image.nii.gz"]
    )
    assert calls[0]["folds"] == (list(range(5)) if model == "mrsegmentator" else [0])
    assert calls[0]["source_profile"] == (
        "mrsegmentator-1.2-base" if model == "mrsegmentator" else "nnunet-v2.8.1-default"
    )
    assert calls[0]["arithmetic"] == ("torch-order-fp32" if model == "mrsegmentator" else "native")
