"""A self-contained local review page rendered from CT voxels and real masks."""

from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

import numpy as np
from nibabel.affines import voxel_sizes
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from medmlx.workflows.ct_contracts import Candidate, Measurement
from medmlx.workflows.ct_geometry import Scan, display_plane, read_nifti


def _png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _frame(
    ct: NDArray[np.float32],
    mask: NDArray[np.float32],
    axis: int,
    index: int,
    spacing: NDArray[np.float64],
    window: tuple[float, float],
    box_xyxy: list[float] | None = None,
) -> list[str]:
    """Render identical CT/mask transforms and preserve physical pixel aspect."""
    plane = display_plane(ct, axis, index)
    selected = display_plane(mask, axis, index)
    grey = np.asarray(
        np.clip((plane - window[0]) / (window[1] - window[0]), 0, 1) * 255, dtype=np.uint8
    )
    axes = [dimension for dimension in range(3) if dimension != axis]
    extent = np.asarray([grey.shape[1] * spacing[axes[0]], grey.shape[0] * spacing[axes[1]]])
    pixels = extent * (360 / extent.max())
    size = (max(1, round(float(pixels[0]))), max(1, round(float(pixels[1]))))
    original = Image.fromarray(grey).resize(size, Image.Resampling.BILINEAR).convert("RGB")
    foreground = Image.fromarray(np.asarray(selected != 0, dtype=np.uint8) * 255).resize(
        size,
        Image.Resampling.NEAREST,
    )
    overlay = Image.blend(original, Image.new("RGB", size, (30, 220, 160)), 0.45)
    combined = Image.composite(overlay, original, foreground)
    if box_xyxy is not None:
        # The box uses the same flipped voxel-centre coordinates as the axial image.
        x0, y0, x1, y1 = box_xyxy
        rectangle = [
            (ct.shape[0] - 0.5 - x1) * size[0] / ct.shape[0],
            (ct.shape[1] - 0.5 - y1) * size[1] / ct.shape[1],
            (ct.shape[0] - 0.5 - x0) * size[0] / ct.shape[0],
            (ct.shape[1] - 0.5 - y0) * size[1] / ct.shape[1],
        ]
        for image in (original, combined):
            ImageDraw.Draw(image).rectangle(rectangle, outline=(255, 209, 93), width=2)
    return [_png(original), _png(combined)]


def write_review(
    scan: Scan,
    candidates: list[Candidate],
    measurements: list[Measurement],
    report: str,
    output: Path,
    window: tuple[float, float],
    *,
    score_threshold: float,
) -> Path:
    """Embed axial slice browsing and orthogonal snapshots, with mask visibility controls.

    Views use the shared RAS working grid, including explicit oblique resampling.
    Downloads contain masks on the original grid. A 30 mm context margin around
    each box keeps small lesions visible; every tracked axial slice is available.
    No network requests, remote assets, model-generated images or raw HTML are used.
    """
    ct = scan.working.get_fdata(dtype=np.float32)
    assert scan.working.affine is not None
    spacing = voxel_sizes(scan.working.affine)
    cards: list[str] = []
    frames: dict[str, list[list[str]]] = {}
    for measurement in measurements:
        candidate = next(c for c in candidates if c.candidate_id == measurement.candidate_id)
        identifier = candidate.candidate_id
        mask = read_nifti(output / identifier / "medsam2/medsam2_mask.nii.gz").get_fdata(
            dtype=np.float32,
        )
        x0, y0, x1, y1 = candidate.box_xyxy
        start, end = candidate.slice_range
        low = np.maximum(np.floor([x0, y0, start]) - np.ceil(30 / spacing), 0).astype(int)
        high = np.minimum(np.ceil([x1, y1, end]) + np.ceil(30 / spacing) + 1, ct.shape).astype(int)
        region = tuple(slice(int(a), int(b)) for a, b in zip(low, high, strict=True))
        crop, labels = ct[region], mask[region]
        centre = np.rint(np.argwhere(labels != 0).mean(axis=0)).astype(int)
        overview = [
            _frame(ct, mask, 2, z, spacing, window, candidate.box_xyxy)
            for z in range(start, end + 1)
        ]
        axial = [
            _frame(crop, labels, 2, z - int(low[2]), spacing, window) for z in range(start, end + 1)
        ]
        coronal = _frame(crop, labels, 1, int(centre[1]), spacing, window)
        sagittal = _frame(crop, labels, 0, int(centre[0]), spacing, window)
        frames[identifier] = axial
        frames[identifier + "-overview"] = overview
        frames[identifier + "-coronal"] = [coronal]
        frames[identifier + "-sagittal"] = [sagittal]
        panels = "".join(
            f'<figure><figcaption>{name}</figcaption><div class="plane">'
            f'<span class="top">{top}</span><span class="left">{left}</span>'
            f'<img id="{identifier}-{key}" src="{pair[1]}" '
            f'alt="{name} CT with mask for {identifier}">'
            f'<span class="right">{right}</span><span class="bottom">{bottom}</span>'
            "</div></figure>"
            for key, name, pair, top, bottom, left, right in (
                (
                    "overview",
                    "CT overview",
                    overview[candidate.frame_idx - start],
                    "A",
                    "P",
                    "R",
                    "L",
                ),
                ("axial", "Axial detail", axial[candidate.frame_idx - start], "A", "P", "R", "L"),
                ("coronal", "Coronal detail", coronal, "S", "I", "R", "L"),
                ("sagittal", "Sagittal detail", sagittal, "S", "I", "A", "P"),
            )
        )
        cards.append(
            f'<section class="candidate" data-id="{identifier}" data-start="{start}">'
            f'<h2>{identifier}</h2><p class="measurements">'
            f"{measurement.volume_mm3:.1f} mm³ · "
            f"{measurement.equivalent_sphere_diameter_mm:.2f} mm volume-equivalent diameter · "
            f"detector score {candidate.score:.3f}</p>"
            '<p class="muted">Mask centroid (RAS mm): '
            + ", ".join(f"{value:.1f}" for value in measurement.centroid_ras_mm)
            + f'</p><div class="views">{panels}</div>'
            f'<div class="controls"><label>Axial slice '
            f'<output id="{identifier}-slice">{candidate.frame_idx}</output>'
            f'<input aria-label="Axial slice for {identifier}" class="slice" type="range" '
            f'min="0" max="{end - start}" value="{candidate.frame_idx - start}"></label>'
            '<label><input class="mask-toggle" type="checkbox" checked> Show mask</label>'
            f'<a href="{identifier}/mask.nii.gz" download>Download source-grid mask</a>'
            "</div></section>"
        )
    if not measurements:
        cards.append('<p class="empty">No candidate masks were produced for this run.</p>')
    payload = json.dumps(frames).replace("<", "\\u003c").replace(">", "\\u003e")
    template = (Path(__file__).with_name("ct_review.html")).read_text()
    context = (
        f"{len(candidates)} candidates offered at score cutoff {score_threshold:.3f}; "
        f"{len(measurements)} segmented. "
        + (
            "Oblique CT resampled for display and prompts."
            if scan.resampled
            else "CT axes reoriented without interpolation."
        )
    )
    substitutions = {
        "__REPORT__": html.escape(report),
        "__SOURCE__": html.escape(scan.source.name),
        "__CARDS__": "".join(cards),
        "__FRAMES__": payload,
        "__CONTEXT__": html.escape(context),
    }
    # Substitute once so text containing template marker names stays literal.
    page = re.sub("|".join(substitutions), lambda match: substitutions[match.group()], template)
    target = output / "review.html"
    target.write_text(page, encoding="utf-8")
    return target
