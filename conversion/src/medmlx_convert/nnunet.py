"""Load the upstream nnU-Net inference network without changing its trainer contract."""

from dataclasses import dataclass
from pathlib import Path

import torch

from medmlx_convert.boundary import as_int, as_ints, attribute, function, library, module
from medmlx_convert.check import CoreMLProfile, export_and_check
from medmlx_convert.guidance import BuildProfile
from medmlx_convert.precision import SourceDevice
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D
from medmlx_convert.runtime_benchmark import Profile


@dataclass(frozen=True)
class NnUNetNetwork:
    model: torch.nn.Module
    input_shape: tuple[int, ...]


def load_nnunet_network(folder: Path, fold: int | str = 0) -> NnUNetNetwork:
    """Use the installed nnU-Net predictor's original checkpoint/trainer loader."""
    predictor_type = function(
        library("nnunetv2.inference.predict_from_raw_data"), "nnUNetPredictor"
    )
    predictor = predictor_type(
        device=torch.device("cpu"), use_mirroring=False, perform_everything_on_device=False
    )
    function(predictor, "initialize_from_trained_model_folder")(str(folder), use_folds=(fold,))
    plans = attribute(predictor, "plans_manager")
    configuration = attribute(predictor, "configuration_manager")
    channels = as_int(function(
        library("nnunetv2.utilities.label_handling.label_handling"),
        "determine_num_input_channels",
    )(plans, configuration, attribute(predictor, "dataset_json")))
    patch = as_ints(attribute(configuration, "patch_size"))
    return NnUNetNetwork(module(predictor, "network"), (1, channels, *patch))


def export_nnunet_folds(
    folder: Path, out_dir: Path, name: str, *, release: str, recipe: Path,
    folds: tuple[int | str, ...] = (0, 1, 2, 3, 4),
    profile: BuildProfile = BuildProfile.FP32, source_device: SourceDevice = SourceDevice.CPU,
    coreml_profile: CoreMLProfile = Profile.COREML_ALL,
) -> bool:
    """Export each declared original fold using its source checkpoint and plans."""
    passed = True
    for fold in folds:
        checkpoint = folder / f"fold_{fold}" / "checkpoint_final.pth"
        network = load_nnunet_network(folder, fold)
        image = torch.randn(network.input_shape, generator=torch.Generator().manual_seed(0))
        passed &= export_and_check(
            network.model, {"image": image}, ["logits"], out_dir / f"{name}_fold_{fold}",
            tolerance=1e-2, coreai_swaps=COREAI_3D, coreml_swaps=COREML_3D,
            source=BuildSource.capture(recipe, checkpoint, origin=release, configuration={
                "plans": (folder / "plans.json").as_posix(),
                "dataset": (folder / "dataset.json").as_posix(),
                "fold": str(fold),
            }),
            profile=profile, source_device=source_device,
            coreml_profile=coreml_profile,
        )
    return passed
