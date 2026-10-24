"""Named released checkpoints executed by the native nnunet-mlx provider."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

PANCREAS = {
    "id": "nnunet-pancreas", "label": "Pancreas segmentation", "task": "segmentation",
    "readiness": "medmlx.adapters.nnunet:readiness",
    "run": "medmlx.adapters.nnunet:pancreas",
    "summary": "PancCTMultiTalentV2 pancreas and cancer segmentation on Metal.",
}
BRATS = {
    "id": "nnunet-brats2021", "label": "nnU-Net BraTS 2021", "task": "segmentation",
    "readiness": "medmlx.adapters.nnunet:readiness",
    "run": "medmlx.adapters.nnunet:brats",
    "summary": "Zenodo 11582627 Dataset002_BRATS19 with its original BAMF labels.",
}
MRSEGMENTATOR = {
    "id": "mrsegmentator", "label": "MRSegmentator", "task": "segmentation",
    "readiness": "medmlx.adapters.nnunet:readiness",
    "run": "medmlx.adapters.nnunet:mrsegmentator",
    "summary": "MRSegmentator 1.2 base weights with native Metal execution.",
}


def _validate_folder(
    model_dir: str, dataset_name: str, channels: dict[str, str],
    labels: dict[str, int] | None = None,
) -> None:
    """Require a converted model folder whose plans and metadata match the named release.

    Checks ``plans.json`` for ``dataset_name`` and ``dataset.json`` for the
    ``channel_names`` order and, when given, the ``labels`` map. Weights are not
    inspected here; ``nnunet-mlx`` reads them when the run starts.

    Raises:
        AssetNotReadyError: A file is missing or unparsable, or the dataset name,
            channel order or label map differs from the expected release.
    """
    from medmlx_core.errors import AssetNotReadyError

    folder = Path(model_dir)
    try:
        plans = json.loads((folder / "plans.json").read_text())
        dataset = json.loads((folder / "dataset.json").read_text())
    except (OSError, ValueError) as error:
        raise AssetNotReadyError(
            f"Select a converted {dataset_name} model folder: {error}"
        ) from error
    if not isinstance(plans, dict) or not isinstance(dataset, dict):
        raise AssetNotReadyError("Model plans and dataset metadata must be objects")
    plans = cast(dict[str, object], plans)
    dataset = cast(dict[str, object], dataset)
    if plans.get("dataset_name") != dataset_name or dataset.get("channel_names") != channels:
        raise AssetNotReadyError(f"Model folder must match {dataset_name} and its channel order")
    if labels is not None and dataset.get("labels") != labels:
        raise AssetNotReadyError(f"Model folder must retain {dataset_name}'s released label map")


def _run(
    model_dir: str, images: list[str], output_dir: str, configuration: str,
    folds: list[int], *, source_profile: str = "nnunet-v2.8.1-default",
    arithmetic: str = "native", use_mirroring: bool = True,
) -> dict[str, object]:
    """Call the ``nnunet-mlx`` runner and expose its ``segmentation`` output as ``mask``."""
    from nnunet_mlx.medmlx import run

    result = run(
        model_dir=model_dir, image_paths=images, output_dir=output_dir,
        configuration=configuration, folds=folds, source_profile=source_profile,
        arithmetic=arithmetic, use_mirroring=use_mirroring,
    )
    return {"outputs": {"mask": result["outputs"]["segmentation"]}, "details": result["details"]}


def pancreas(
    *, model_dir: str, image_path: str, output_dir: str, fold: int = 0,
    use_mirroring: bool = True,
) -> dict[str, object]:
    """Segment pancreas and cancer in one CT with the converted PancCTMultiTalentV2 model.

    Only the inference-staged ``Dataset999_PancCTPretrain`` checkpoint
    (``3d_fullres_bs4``) is accepted, not other CT nnU-Net models.

    Args:
        model_dir: Converted nnU-Net model folder holding ``plans.json`` and
            ``dataset.json`` for that dataset (labels: background 0, pancreas 1,
            cancer 2).
        image_path: Path of the input CT volume.
        output_dir: Directory for the predicted mask.
        fold: Trained fold to run.
        use_mirroring: Whether to average over test-time flips.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}`` from ``nnunet-mlx``.

    Raises:
        AssetNotReadyError: ``model_dir`` is not the expected converted folder.
    """
    _validate_folder(
        model_dir, "Dataset999_PancCTPretrain", {"0": "CT"},
        {"background": 0, "pancreas": 1, "cancer": 2},
    )
    return _run(
        model_dir, [image_path], output_dir, "3d_fullres_bs4", [fold],
        use_mirroring=use_mirroring,
    )


def brats(
    *, model_dir: str, flair_path: str, t1_path: str, t1ce_path: str, t2_path: str,
    output_dir: str, fold: int = 0, use_mirroring: bool = True,
) -> dict[str, object]:
    """Segment a BraTS study with the converted ``Dataset002_BRATS19`` nnU-Net model.

    The four co-registered modality paths are passed to the provider in the
    released channel order (T1, T1ce, T2, FLAIR) whatever the keyword order, and
    the output keeps the released BAMF labels (background 0, edema 1,
    nonenhancing 2, empty 3, enhancing 4).

    Args:
        model_dir: Converted model folder with matching ``plans.json`` and
            ``dataset.json``.
        flair_path, t1_path, t1ce_path, t2_path: Paths of the four input volumes.
        output_dir: Directory for the predicted mask.
        fold: Trained fold, 0 to 4.
        use_mirroring: Whether to average over test-time flips.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}`` from ``nnunet-mlx``.

    Raises:
        InvalidInputError: ``fold`` is outside 0 to 4.
        AssetNotReadyError: ``model_dir`` is not the expected converted folder.
    """
    from medmlx_core.errors import InvalidInputError

    if fold not in range(5):
        raise InvalidInputError("BraTS fold must be between 0 and 4")
    _validate_folder(
        model_dir, "Dataset002_BRATS19", {"0": "t1", "1": "t1ce", "2": "t2", "3": "flair"},
        {"background": 0, "edema": 1, "nonenhancing": 2, "empty": 3, "enhancing": 4},
    )
    images = [t1_path, t1ce_path, t2_path, flair_path]
    return _run(
        model_dir, images, output_dir, "3d_fullres", [fold], use_mirroring=use_mirroring,
    )


def mrsegmentator(
    *, model_dir: str, image_path: str, output_dir: str, fast: bool = False,
    fold: int | None = None,
) -> dict[str, object]:
    """Segment one MR volume with the converted MRSegmentator 1.2 base model.

    With no ``fold``, ``fast=True`` selects fold 0 and the default uses all five
    folds. An explicit fold takes precedence over ``fast``. The provider is
    called with the MRSegmentator profile, Torch-order FP32 arithmetic, and
    mirroring disabled for every selection.

    Args:
        model_dir: Converted ``Dataset050_ukbb_inhouse_nako`` model folder.
        image_path: Path of the input MR volume.
        output_dir: Directory for the predicted mask.
        fast: Run fold 0 only when ``fold`` is not given.
        fold: Single fold to run, 0 to 4.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}`` from ``nnunet-mlx``.

    Raises:
        InvalidInputError: ``fold`` is outside 0 to 4.
        AssetNotReadyError: ``model_dir`` is not the expected converted folder.
    """
    from medmlx_core.errors import InvalidInputError

    if fold is not None and fold not in range(5):
        raise InvalidInputError("MRSegmentator fold must be between 0 and 4")
    _validate_folder(model_dir, "Dataset050_ukbb_inhouse_nako", {"0": "MR"})
    folds = [fold] if fold is not None else [0] if fast else list(range(5))
    return _run(
        model_dir, [image_path], output_dir, "3d_fullres_bs8", folds,
        source_profile="mrsegmentator-1.2-base", arithmetic="torch-order-fp32",
        use_mirroring=False,
    )


def readiness() -> dict[str, object]:
    """Require the independent nnU-Net runtime; model folders are checked per run."""
    try:
        version("nnunet-mlx")
    except PackageNotFoundError:
        return {"ready": False, "reason": "nnunet-mlx is not installed",
                "setup_hint": "Install the segment extra: uv sync --locked --extra segment"}
    return {"ready": True, "reason": None, "setup_hint": None}
