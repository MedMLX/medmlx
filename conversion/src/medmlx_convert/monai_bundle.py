"""Build a MONAI bundle's network the way the bundle's own config does."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from monai.bundle.config_parser import ConfigParser

from medmlx_convert.boundary import as_state_dict, function, hf_snapshot, unwrap


@dataclass(frozen=True)
class BundleNetwork:
    model: torch.nn.Module
    checkpoint: Path
    origin: str
    configuration: dict[str, str]


def load_bundle_network(
    repo_id: str,
    revision: str,
    network_key: str = "network_def",
    checkpoint: str = "models/model.pt",
    drop_args: tuple[str, ...] = (),
    set_args: dict[str, object] | None = None,
) -> BundleNetwork:
    """Build a MONAI bundle at its declared revision through its own config loader.

    ``drop_args`` removes network arguments that the pinned MONAI no longer accepts;
    ``set_args`` overrides network arguments.
    """
    root = hf_snapshot(
        repo_id, revision, ["configs/*", "scripts/*", "scripts/**/*", checkpoint]
    )
    # Bundles with custom networks reference them as `scripts.<module>`.
    if (root / "scripts").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    parser = ConfigParser()
    function(parser, "read_config")(str(root / "configs" / "inference.json"))
    parser["bundle_root"] = str(root)
    for name in drop_args:
        del parser[network_key][name]
    for name, value in (set_args or {}).items():
        parser[f"{network_key}#{name}"] = value
    network = parser.get_parsed_content(network_key)
    assert isinstance(network, torch.nn.Module)
    loaded: object = torch.load(root / checkpoint, map_location="cpu", weights_only=True)
    # Some bundles wrap the weights as {"model": state_dict}.
    state = unwrap(loaded, "model")
    network.load_state_dict(as_state_dict(state, str(root / checkpoint)))
    return BundleNetwork(network.eval(), root / checkpoint, f"{repo_id}@{revision}", {
        "config": "configs/inference.json", "network_key": network_key,
        **{f"drop/{name}": "" for name in drop_args},
        **{f"set/{name}": str(value) for name, value in (set_args or {}).items()},
    })
