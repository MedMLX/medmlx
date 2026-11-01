"""Export every released AutoPET II fold from the original nnU-Net plans."""

import sys
from pathlib import Path

from medmlx_convert import recipe_args
from medmlx_convert.nnunet import export_nnunet_folds
from medmlx_convert.sources import checkpoint_archive

options = recipe_args("nnunet_autopet")
out_dir = options.out
URL = (
    "https://zenodo.org/api/records/8362371/files/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres_resenc_bs80_exported.zip/content"
)
root = checkpoint_archive(
    URL,
    "0c60a713c681ac5a7b134f36ba3d85814702cd0c746b5a15639e1e899331cc7c",
)
passed = export_nnunet_folds(
    root / "Dataset221_AutoPETII_2023"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres_resenc_bs80",
    out_dir, "nnunet_autopet", release=URL, recipe=Path(__file__),
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
