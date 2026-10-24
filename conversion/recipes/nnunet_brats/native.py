"""Convert the released BraTS fold 0 network at a declared diagnostic patch size."""

import argparse
from pathlib import Path

import torch

from medmlx_convert.native.check import convert_and_check
from medmlx_convert.nnunet import load_nnunet_network
from medmlx_convert.provenance import BuildSource

RELEASE = "https://zenodo.org/api/records/11582627/files/Dataset002_BRATS19.zip/content"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted release root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=64,
                        help="Cubic network-only diagnostic patch; not the planned image pipeline")
    options = parser.parse_args()
    if options.patch < 64 or options.patch % 32:
        raise ValueError("BraTS diagnostic patch must be a multiple of 32, at least 64")
    folder = (options.root / "Dataset002_BRATS19"
              / "nnUNetTrainer__nnUNetPlans__3d_fullres")
    checkpoint = folder / "fold_0/checkpoint_final.pth"
    network = load_nnunet_network(folder, 0)
    network.model.eval()
    image = torch.randn((1, network.input_shape[1], *((options.patch,) * 3)),
                         generator=torch.Generator().manual_seed(0))
    report = convert_and_check(network.model, (image,), options.out,
                               source=BuildSource.capture(
                                   Path(__file__), checkpoint, origin=RELEASE,
                                   configuration={"plans": (folder / "plans.json").as_posix(),
                                                  "dataset": (folder / "dataset.json").as_posix(),
                                                  "fold": "0"},
                               ), tolerance=1e-2)
    for comparison in report.comparisons:
        print(f"{comparison.framework}: max_abs_diff={comparison.max_abs_diff:.8g}", flush=True)


if __name__ == "__main__":
    main()
