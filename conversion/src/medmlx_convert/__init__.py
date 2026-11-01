"""Shared export, check, and rewrite helpers for the recipes."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medmlx_convert.check import export_and_check, recipe_args
    from medmlx_convert.monai_bundle import load_bundle_network
    from medmlx_convert.rewrites import (
        FlatInstanceNorm3d,
        StridedTransposedConv3d,
        replace_modules,
    )

__all__ = [
    "FlatInstanceNorm3d",
    "StridedTransposedConv3d",
    "export_and_check",
    "load_bundle_network",
    "recipe_args",
    "replace_modules",
]


def __getattr__(name: str) -> object:
    """Load export tools on demand so native MLX inference does not import Torch."""
    owners = {
        "export_and_check": "check", "recipe_args": "check",
        "load_bundle_network": "monai_bundle", "FlatInstanceNorm3d": "rewrites",
        "StridedTransposedConv3d": "rewrites", "replace_modules": "rewrites",
    }
    if name not in owners:
        raise AttributeError(name)
    return getattr(import_module(f"medmlx_convert.{owners[name]}"), name)
