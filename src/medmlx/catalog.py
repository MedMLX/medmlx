"""Checkpoint source references, separate from installed runners and qualification.

Named entries follow adapters/nnunet.py. Liver and AutoPET source records match
RadNN's nnunet_liver and gc_autopet_fpr catalogs at revision
55db0359be8295ca6b88a5031568e669eba0b9dd. They remain catalog-only references:
this module neither stages weights nor asserts that a checkpoint was qualified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Checkpoint:
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
        "Supply the inference-staged, converted model_dir with 3d_fullres_bs4 plans; "
        "retain background/pancreas/cancer labels. Upstream fold_all needs inference staging.",
    ),
    Checkpoint(
        "brats19-zenodo-11582627", "BraTS19 released BAMF checkpoint", "nnunet-brats2021",
        "https://zenodo.org/records/11582627", "Dataset002_BRATS19",
        ("T1", "T1ce", "T2", "FLAIR"), "named_adapter",
        "Supply converted model_dir with 3d_fullres plans and the released BAMF labels, "
        "plus four co-registered NIfTI inputs.",
    ),
    Checkpoint(
        "mrsegmentator-1.2-base", "MRSegmentator 1.2 base", "mrsegmentator",
        "https://github.com/hhaentze/MRSegmentator/releases/tag/v1.2.0",
        "Dataset050_ukbb_inhouse_nako", ("MR",), "named_adapter",
        "Supply converted 1.2 base model_dir with 3d_fullres_bs8 plans. The named adapter "
        "selects mrsegmentator-1.2-base, torch-order-fp32 and no mirroring. Verify weight terms.",
    ),
    Checkpoint(
        "liver-zenodo-18807203", "CT liver and liver tumor", "nnunet",
        "https://zenodo.org/records/18807203", "Dataset003_Liver", ("CT (HU)",), "catalog_only",
        "Source reference only. Convert and validate the upstream nnU-Net folder before "
        "supplying model_dir to the general nnunet runner; select configuration from its plans. "
        "No named MedMLX adapter or checkpoint qualification is implied.",
    ),
    Checkpoint(
        "autopet-ii-zenodo-8362371", "AutoPET II PET/CT tumor", "nnunet",
        "https://zenodo.org/records/8362371", "Dataset221_AutoPETII_2023",
        ("CT (HU)", "FDG PET (SUV)"), "catalog_only",
        "Source reference: bs80 exported model. Conversion and checkpoint validation are "
        "required before using model_dir with the general nnunet runner. Supply co-registered "
        "image_paths in CT then SUV PET order. No checkpoint qualification is implied.",
    ),
)
