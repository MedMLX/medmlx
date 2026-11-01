"""A bounded Reason-CT agent with spatially grounded detector/MedSAM2 tools."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from inspect import signature
from pathlib import Path

#: Each weight argument and the model whose automatic weights it uses.
WEIGHTS = {
    "reason_weights_path": "mlx-reason-ct",
    "detector_weights_path": "lung-nodule",
    "medsam_weights_path": "medsam2",
}

MODEL = {
    "id": "reason-ct-tools",
    "label": "Reason-CT nodule tools",
    "task": "workflow",
    "run": "medmlx.workflows.reason_ct:run",
    "readiness": "medmlx.workflows.reason_ct:readiness",
    "weights_from": WEIGHTS,
    "summary": "Reason-CT calls nodule detection and MedSAM2; source-grid masks and review HTML.",
}


def readiness() -> dict[str, object]:
    """Check imports and the required provider API without initializing a model."""
    missing = [
        name
        for name in (
            "mlx_reason_ct",
            "lung_nodule_mlx",
            "medsam2_mlx",
            "nibabel",
            "scipy",
            "PIL",
            "pydantic",
            "llguidance",
        )
        if find_spec(name) is None
    ]
    if missing:
        return {
            "ready": False,
            "reason": f"Missing {', '.join(missing)}",
            "setup_hint": "Install this source checkout with its reason-tools extra.",
        }
    from mlx_reason_ct.medmlx import run as reason_run

    supported = {"tools", "tool_history", "response_choices"} <= set(
        signature(reason_run).parameters
    )
    return {
        "ready": supported,
        "reason": None if supported else "Reason-CT provider lacks the tool conversation API",
        "setup_hint": (
            None if supported else "Install this source checkout with its reason-tools extra."
        ),
    }


def run(
    *,
    image_path: str | Path,
    output_dir: str | Path,
    reason_weights_path: str | Path | None = None,
    detector_weights_path: str | Path | None = None,
    medsam_weights_path: str | Path | None = None,
    prompt: str = "Find lung nodule candidates, segment them, and show the masks and measurements.",
    score_threshold: float = 0.25,
    max_candidates: int = 5,
    max_segments: int = 2,
    max_steps: int = 8,
    max_new_tokens: int = 512,
    padding_mm: float = 3.0,
    hu_window: Sequence[float] = (-1000.0, 400.0),
    assume_mm: bool = False,
) -> dict[str, object]:
    """Run a chest-CT tool conversation inside the standard supervised worker.

    Uses converted FP32 Reason-CT, RetinaNet and MedSAM2 CT-lesion weights. The
    detector orders tied scores by original index; overlapping boxes can
    differ from the upstream MONAI output. The model may request detection,
    segment a returned candidate ID, or finish. LLGuidance constrains each
    decision to the native calls currently allowed by the workflow. It cannot
    supply paths, spatial coordinates, code or model options. Invalid requests
    receive explicit error feedback and consume a step; exhaustion fails with
    the partial trace retained.

    Args:
        image_path: Finite scalar 3D chest CT NIfTI with HU intensities and a coded,
            unambiguous spatial transform. All tools receive the same RAS/mm CT.
        output_dir: New directory only; existing results are never replaced.
        reason_weights_path: Reason-CT model folder. Omit to use the
            ``mlx-reason-ct`` automatic download.
        detector_weights_path: Lung-nodule model folder or ``network.safetensors``.
            Omit to use the ``lung-nodule`` automatic download.
        medsam_weights_path: MedSAM2 model folder or ``MedSAM2_CTLesion.safetensors``.
            Omit to use the ``medsam2`` automatic download.
        score_threshold: Candidate score cutoff, not a calibrated probability.
        max_candidates: Highest-scoring candidates offered to Reason-CT.
        max_segments: Maximum number of distinct candidates the model may segment.
        padding_mm: Extend the detector box's z interval for bidirectional tracking.
        hu_window: Explicit lower/upper HU bounds shared by MedSAM2 and the viewer.
        assume_mm: Explicitly interpret unknown NIfTI spatial units as millimetres.

    Returns:
        Output paths for ``review.html``, ``report.txt`` and ``run.json``. The run
        record contains actions, tool results, source geometry and mask-derived
        measurements. Each candidate has a binary NIfTI mask on the original grid.

    Raises:
        ValueError: Inputs, geometry, masks or limits violate the workflow contract.
        RuntimeError: A provider fails or the model cannot complete within max_steps.
        FileExistsError: output_dir already exists.
    """
    from medmlx.workflows.ct_agent import execute
    from medmlx.workflows.ct_contracts import Limits

    limits = Limits(
        prompt=prompt,
        score_threshold=score_threshold,
        max_candidates=max_candidates,
        max_segments=max_segments,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        padding_mm=padding_mm,
        assume_mm=assume_mm,
    )
    return execute(
        image_path=Path(image_path),
        reason_weights_path=_weights("reason_weights_path", reason_weights_path),
        detector_weights_path=_weights("detector_weights_path", detector_weights_path),
        medsam_weights_path=_weights("medsam_weights_path", medsam_weights_path),
        output_dir=Path(output_dir),
        limits=limits,
        hu_window=hu_window,
    )


def _weights(argument: str, value: str | Path | None) -> Path:
    """Use an explicit path as given; otherwise the model's automatic download."""
    if value is not None:
        return Path(value)
    from medmlx_core.weights import weights_dir

    return weights_dir(WEIGHTS[argument])
