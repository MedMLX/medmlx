"""Export all five original Zenodo BraTS folds, preserving its BRATS19 archive name."""

import sys
from pathlib import Path

from medmlx_convert import recipe_args
from medmlx_convert.nnunet import export_nnunet_folds
from medmlx_convert.sources import checkpoint_archive

URL = "https://zenodo.org/api/records/11582627/files/Dataset002_BRATS19.zip/content"
ARCHIVE_CACHE_KEY = "f45eff0624604333a80ca5e6ea9c71153d908a4c2082df9038051dd7c4b2491f"

options = recipe_args("nnunet_brats")
out_dir = options.out
passed = export_nnunet_folds(
    checkpoint_archive(URL, ARCHIVE_CACHE_KEY) / "Dataset002_BRATS19"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres", out_dir, "nnunet_brats",
    release=URL, recipe=Path(__file__),
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
