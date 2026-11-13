"""Load the upstream nnU-Net inference network without changing its trainer contract."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from medmlx_core import load_torch_checkpoint
from numpy._core.multiarray import scalar as numpy_scalar

from medmlx_convert.boundary import (
    as_int,
    as_ints,
    as_state_dict,
    as_str_dict,
    attribute,
    function,
    library,
    module,
)
from medmlx_convert.check import CoreMLProfile, export_and_check
from medmlx_convert.guidance import BuildProfile
from medmlx_convert.image_inputs import ct_tensor
from medmlx_convert.precision import SourceDevice
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D
from medmlx_convert.runtime_benchmark import Profile


@dataclass(frozen=True)
class NnUNetNetwork:
    """An admitted nnU-Net network with its fixed input shape and configuration name."""

    model: torch.nn.Module
    input_shape: tuple[int, ...]
    configuration: str


def load_nnunet_network(folder: Path, fold: int | str = 0) -> NnUNetNetwork:
    """Load an admitted checkpoint through nnU-Net's original predictor contract.

    The pinned upstream loader uses unrestricted pickle deserialization. Admit
    the original with Torch's restricted loader, retain only inference fields,
    and give upstream a temporary checkpoint serialized from those safe values.
    """
    fold_name = "all" if fold == "all" else str(int(fold))
    checkpoint = folder / f"fold_{fold_name}" / "checkpoint_final.pth"
    # Qualified nnU-Net checkpoints contain only NumPy FP32/FP64 metadata beyond
    # Torch's default safe set. Accept those data constructors under both NumPy
    # module paths used by the pinned source releases; all other globals fail closed.
    with torch.serialization.safe_globals([
        numpy_scalar,
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]):
        payload = load_torch_checkpoint(checkpoint)
    trainer_name = payload.get("trainer_name")
    if not isinstance(trainer_name, str) or not trainer_name:
        raise TypeError("nnU-Net checkpoint trainer_name must be non-empty text")
    init_args = as_str_dict(payload.get("init_args"), "nnU-Net checkpoint init_args")
    configuration = init_args.get("configuration")
    if not isinstance(configuration, str) or not configuration:
        raise TypeError("nnU-Net checkpoint configuration must be non-empty text")
    network_weights = as_state_dict(
        payload.get("network_weights"), "nnU-Net checkpoint network_weights"
    )
    raw_axes = payload.get("inference_allowed_mirroring_axes")
    mirroring_axes = (
        None if raw_axes is None else as_ints(raw_axes, "inference_allowed_mirroring_axes")
    )
    predictor_type = function(
        library("nnunetv2.inference.predict_from_raw_data"), "nnUNetPredictor"
    )
    predictor = predictor_type(
        device=torch.device("cpu"), use_mirroring=False, perform_everything_on_device=False
    )
    with TemporaryDirectory(prefix="medmlx-nnunet-") as directory:
        admitted = Path(directory)
        shutil.copyfile(folder / "plans.json", admitted / "plans.json")
        shutil.copyfile(folder / "dataset.json", admitted / "dataset.json")
        admitted_fold = admitted / f"fold_{fold_name}"
        admitted_fold.mkdir()
        torch.save(
            {
                "trainer_name": trainer_name,
                "init_args": {"configuration": configuration},
                "network_weights": network_weights,
                "inference_allowed_mirroring_axes": mirroring_axes,
            },
            admitted_fold / "checkpoint_final.pth",
        )
        function(predictor, "initialize_from_trained_model_folder")(
            str(admitted), use_folds=(fold_name,)
        )
    plans = attribute(predictor, "plans_manager")
    configuration_manager = attribute(predictor, "configuration_manager")
    channels = as_int(function(
        library("nnunetv2.utilities.label_handling.label_handling"),
        "determine_num_input_channels",
    )(plans, configuration_manager, attribute(predictor, "dataset_json")))
    patch = as_ints(attribute(configuration_manager, "patch_size"))
    return NnUNetNetwork(module(predictor, "network"), (1, channels, *patch), configuration)


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
        image = ct_tensor(network.input_shape)
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
