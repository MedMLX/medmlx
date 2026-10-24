"""Export the released PANTHER pancreas fold_all with its unchanged network weights.

The pinned PANTHER source's nnUNetTrainer1e3 only changes initial_lr; its inference
builder is inherited from nnUNetTrainer. The lookup below makes that choice explicit
without editing checkpoint metadata or accepting any other trainer identity.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from medmlx_convert import recipe_args
from medmlx_convert.boundary import class_, library
from medmlx_convert.nnunet import export_nnunet_folds
from medmlx_convert.sources import checkpoint_archive


def inference_trainer(name: str) -> type:
    if name != "nnUNetTrainer1e3":
        raise ValueError(f"Unexpected PANTHER checkpoint trainer: {name}")
    return class_(library("nnunetv2.training.nnUNetTrainer.nnUNetTrainer"), "nnUNetTrainer")


options = recipe_args("nnunet_pancreas")
out_dir = options.out
URL = 'https://zenodo.org/api/records/17212158/files/PancCTMultiTalentV2.zip/content'
root = checkpoint_archive(
    URL,
    "cf44c737ae2f988dcd504df9af02e26ba147c3e1b351c1c5cf56a4a34fcfa529",
)
with patch(
    "nnunetv2.inference.predict_from_raw_data.recursive_find_trainer_class_by_name",
    new=inference_trainer,
):
    passed = export_nnunet_folds(
        root / "PancCTMultiTalentV2/Dataset999_PancCTPretrain"
        / "nnUNetTrainer1e3__nnUNetResEncUNetLPlans__3d_fullres_bs4",
        out_dir, "nnunet_pancreas", release=URL, recipe=Path(__file__), folds=("all",),
        profile=options.profile, source_device=options.source_device,
        coreml_profile=options.coreml_profile,
    )
sys.exit(0 if passed else 1)
