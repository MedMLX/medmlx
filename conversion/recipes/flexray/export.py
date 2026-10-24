"""Export all pinned FleXray ensemble members and check both formats against PyTorch."""

import sys
import types
import urllib.request
from importlib import import_module
from pathlib import Path

import torch
import yaml

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    as_module,
    as_str_dict,
    hf_snapshot,
    load_safetensors,
    module_class,
)
from medmlx_convert.evidence import LabelMode, LabelRequirement
from medmlx_convert.provenance import BuildSource

REPO = "VictorButoi/flexray"
REVISION = "6b63ddf09cfdff99aea494e3738cf243a4e22cdb"
# The FleXray package's own uv config points torch at a CUDA index, so only the model source
# is fetched, at the commit flexray-mlx pins.
SOURCE_REVISION = "bf6d01426fd3ae012aff9253ec9c9952c58bc960"
SOURCE_FILES = ("models/unet.py", "models/nn/__init__.py", "models/nn/blocks.py")


def fxr_unet() -> type[torch.nn.Module]:
    root = Path(torch.hub.get_dir()) / "flexray" / SOURCE_REVISION / "fxr"
    base = f"https://raw.githubusercontent.com/VictorButoi/FleXray/{SOURCE_REVISION}/src/fxr"
    for name in SOURCE_FILES:
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{base}/{name}", path)
    # Namespace parents skip fxr's training-only package imports.
    for package, directory in (("fxr", root), ("fxr.models", root / "models")):
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    return module_class(import_module("fxr.models.unet"), "UNet")


options = recipe_args("flexray")
out_dir = options.out
snapshot = hf_snapshot(
    REPO, REVISION, ["members/*/config.yml", "members/*/model.safetensors"]
)
# flexray-mlx runs one 256x256 radiograph after percentile min-max scaling to [0, 1].
image = torch.rand((1, 1, 256, 256), generator=torch.Generator().manual_seed(0))
model_class = fxr_unet()
passed = True
for member_name in ("flux000", "flux025", "flux0375", "flux050", "flux075"):
    member = snapshot / "members" / member_name
    config_path = member / "config.yml"
    document = as_str_dict(yaml.safe_load(config_path.read_text()), "config.yml")
    config = as_str_dict(document["model"], "config.yml model")
    if config.pop("_class") != "fxr.models.UNet":
        raise ValueError(f"unexpected FleXray model class in {config_path}")
    model = as_module(model_class(**config), "FleXray UNet")
    model.load_state_dict(load_safetensors(member / "model.safetensors"))
    # Independent sigmoid labels: argmax agreement is not a segmentation metric here.
    passed &= export_and_check(
        model, {"image": image}, ["multilabel_logits"],
        out_dir / f"flexray_{member.name}", tolerance=1e-2,
        source=BuildSource.capture(
            Path(__file__), member / "model.safetensors", origin=f"{REPO}@{REVISION}",
            configuration={"config": config_path.as_posix(),
                           "source": f"VictorButoi/FleXray@{SOURCE_REVISION}"},
        ),
        profile=options.profile, source_device=options.source_device,
        coreml_profile=options.coreml_profile,
        # Each channel is an independent sigmoid decision at 0.5 (raw logit >0).
        output_requirements=(LabelRequirement("multilabel_logits", LabelMode.INDEPENDENT, 0.999),),
    )
sys.exit(0 if passed else 1)
