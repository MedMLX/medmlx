"""Checkpoint source references, separate from installed runners and qualification.

Named entries follow nnunet_mlx.named_models. Liver and AutoPET source records match
RadNN's nnunet_liver and gc_autopet_fpr catalogs at revision
55db0359be8295ca6b88a5031568e669eba0b9dd. These are source references only:
this module neither downloads weights nor asserts that a checkpoint was qualified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Source metadata and intended runner; an entry does not qualify its checkpoint."""

    id: str
    label: str
    model_id: str
    source_url: str
    dataset: str
    input_channels: tuple[str, ...]
    integration: Literal["named_adapter", "catalog_only"]
    asset_hint: str


CHECKPOINTS = (
    Checkpoint(
        "pancct-multitalent-v2", "PancCTMultiTalentV2", "nnunet-pancreas",
        "https://zenodo.org/records/17212158", "Dataset999_PancCTPretrain", ("CT (HU)",),
        "named_adapter",
        "Optional local folder: 3d_fullres_bs4 inference-staged plans; "
        "retain background/pancreas/cancer labels. Upstream fold_all needs inference staging.",
    ),
    Checkpoint(
        "brats19-zenodo-11582627", "BraTS19 released BAMF checkpoint", "nnunet-brats2021",
        "https://zenodo.org/records/11582627", "Dataset002_BRATS19",
        ("T1", "T1ce", "T2", "FLAIR"), "named_adapter",
        "Optional local folder: 3d_fullres plans and the released BAMF labels, "
        "plus four co-registered NIfTI inputs.",
    ),
    Checkpoint(
        "mrsegmentator-1.2-base", "MRSegmentator 1.2 base", "mrsegmentator",
        "https://github.com/hhaentze/MRSegmentator/releases/tag/v1.2.0",
        "Dataset050_ukbb_inhouse_nako", ("MR",), "named_adapter",
        "Optional local 1.2 base folder with 3d_fullres_bs8 plans. The named adapter "
        "selects mrsegmentator-1.2-base, torch-order-fp32 and no mirroring. Verify weight terms.",
    ),
    Checkpoint(
        "liver-zenodo-18807203", "CT liver and liver tumor", "nnunet-liver",
        "https://zenodo.org/records/18807203", "Dataset003_Liver", ("CT (HU)",), "named_adapter",
        "Omit weights_path to download the named liver model automatically; "
        "a local model folder can override it. No checkpoint qualification is implied.",
    ),
    Checkpoint(
        "autopet-ii-zenodo-8362371", "AutoPET II PET/CT tumor", "nnunet-autopet",
        "https://zenodo.org/records/8362371", "Dataset221_AutoPETII_2023",
        ("CT (HU)", "FDG PET (SUV)"), "named_adapter",
        "Omit weights_path to download the named AutoPET model automatically. Supply "
        "co-registered image_paths in CT then SUV PET order. The runner selects bs80 plans; "
        "no checkpoint qualification is implied.",
    ),
)
