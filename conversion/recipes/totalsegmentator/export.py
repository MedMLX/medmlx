"""Export TotalSegmentator's 3 mm model to Core AI and Core ML, then check both against PyTorch."""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.nnunet import load_nnunet_network
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

# TotalSegmentator v2 `--fast`: all 117 classes of the open `total` task (Apache-2.0) at 3 mm,
# as one nnU-Net PlainConvUNet. nnunet-mlx runs it under the totalsegmentator-v2-ct profile.
DATASET = "Dataset297_TotalSegmentator_total_3mm_1559subj"
TRAINED_MODEL = "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
WEIGHTS_URL = (
    f"https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/{DATASET}.zip"
)


def weights_dir() -> Path:
    """Use explicit TotalSeg paths or a converter-owned upstream source cache."""
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        return Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    # Other ports rewrite TotalSegmentator's plans in place. Keep the source oracle separate.
    default_home = Path(torch.hub.get_dir()) / "medmlx-convert" / "totalsegmentator-v2.0.0"
    home = Path(os.environ.get("TOTALSEG_HOME_DIR", default_home))
    return home / "nnunet" / "results"


def trained_model_folder() -> Path:
    results = weights_dir()
    folder = results / DATASET / TRAINED_MODEL
    if not folder.exists():
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / f"{DATASET}.zip"
            torch.hub.download_url_to_file(WEIGHTS_URL, str(archive))
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(results)
    return folder


options = recipe_args("totalsegmentator")
out_dir = options.out
folder = trained_model_folder()
network = load_nnunet_network(folder)

# One sliding-window patch at the plans patch size, after nnU-Net's CT normalization.
image = torch.randn(network.input_shape, generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    network.model,
    {"image": image},
    ["logits"],
    out_dir / "totalsegmentator_3mm",
    source=BuildSource.capture(
        Path(__file__), folder / "fold_0/checkpoint_final.pth", origin=WEIGHTS_URL,
        configuration={"plans": (folder / "plans.json").as_posix(),
                       "dataset": (folder / "dataset.json").as_posix()},
    ), tolerance=1e-2,
    coreai_swaps=COREAI_3D,
    coreml_swaps=COREML_3D,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
