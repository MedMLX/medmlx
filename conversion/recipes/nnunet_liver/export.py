"""Export the released Zenodo 18807203 liver nnU-Net at its planned patch size."""

import sys
from pathlib import Path

import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.nnunet import load_nnunet_network
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

RECORD = "https://zenodo.org/api/records/18807203/files"

options = recipe_args("nnunet_liver")
out_dir = options.out
folder = Path(torch.hub.get_dir()) / "nnunet-liver-18807203"
for filename in ("plans.json", "dataset.json", "checkpoint_final.pth"):
    target = folder / ("fold_0" if filename.endswith(".pth") else "") / filename
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(f"{RECORD}/{filename}/content", str(target))
network = load_nnunet_network(folder)
image = torch.randn(network.input_shape, generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    network.model, {"image": image}, ["logits"], out_dir / "nnunet_liver",
    source=BuildSource.capture(
        Path(__file__), folder / "fold_0/checkpoint_final.pth", origin=RECORD,
        configuration={"plans": (folder / "plans.json").as_posix(),
                       "dataset": (folder / "dataset.json").as_posix()},
    ), tolerance=1e-2,
    coreai_swaps=COREAI_3D, coreml_swaps=COREML_3D,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
