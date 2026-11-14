"""Execute native Qwen-template tool requests under a bounded CT workflow policy."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from medmlx import registry
from medmlx.payloads import JsonValue
from medmlx.workflows.ct_contracts import (
    Action,
    Candidate,
    Detect,
    Finish,
    Limits,
    Measurement,
    Segment,
)
from medmlx.workflows.ct_geometry import ground_candidates, prepare_scan, restore_mask
from medmlx.workflows.ct_viewer import write_review

TOOLS: list[dict[str, JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "detect_nodules",
            "description": "Detect nodule candidates. Call once, before segmentation.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_candidate",
            "description": "Segment one returned candidate ID with MedSAM2 and measure its mask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "An ID returned by detect_nodules.",
                    },
                },
                "required": ["candidate_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish tool use and prepare the review and interpretation.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def tool_choice(name: str, candidate_id: str | None = None) -> str:
    """Render the checkpoint's native call syntax from host-owned choices."""
    parameter = (
        "" if candidate_id is None else f"<parameter=candidate_id>\n{candidate_id}\n</parameter>\n"
    )
    return f"<tool_call>\n<function={name}>\n{parameter}</function>\n</tool_call>"


def parse_action(text: str) -> Action:
    """Read the pinned Qwen3.5 XML-style format for the workflow's supported functions.

    The template permits prose before a single call, but no suffix. Parameters
    are raw text, not XML attributes or JSON; only the candidate ID is admitted.
    Multiple calls, extra parameters and malformed tags never reach a runner.
    Final interpretation is generated separately after the finish action.
    """
    if not any(marker in text for marker in ("<tool_call", "<function=", "<parameter=")):
        raise ValueError("A complete tool call is required for an action decision")
    match = re.fullmatch(
        r"(?s).*?<tool_call>\s*<function=([A-Za-z_]\w*)>\s*(.*?)\s*</function>"
        r"\s*</tool_call>\s*",
        text,
    )
    if match is None or text.count("<tool_call>") != 1:
        raise ValueError("Return one complete native tool call, with no suffix or extra calls")
    name, arguments = match.groups()
    if name == "detect_nodules" and not arguments.strip():
        return Detect(action="detect_nodules")
    if name == "finish" and not arguments.strip():
        return Finish(action="finish")
    if name == "segment_candidate":
        parameter = re.fullmatch(
            r"<parameter=candidate_id>\s*(candidate-[0-9]{3,})\s*</parameter>",
            arguments,
        )
        if parameter is not None:
            return Segment(action="segment_candidate", candidate_id=parameter.group(1))
    raise ValueError("Unsupported tool or arguments; use detection, a candidate ID, or finish")


def _write_record(path: Path, record: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def execute(
    *,
    image_path: Path,
    reason_weights_path: Path,
    detector_weights_path: Path,
    medsam_weights_path: Path,
    output_dir: Path,
    limits: Limits,
    hu_window: Sequence[float],
) -> dict[str, object]:
    """Run providers serially inside the caller's supervised worker/process group.

    The standard API/CLI/MCP worker owns Metal admission and cancellation. No
    nested detached jobs are launched. Model text is only parsed as a bounded
    request; all paths, transforms, model IDs and measurements are host-owned.
    """
    window = np.asarray(hu_window, dtype=np.float64)
    if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError("hu_window must contain finite increasing lower/upper HU bounds")
    paths = [
        path.expanduser().resolve()
        for path in (
            image_path,
            reason_weights_path,
            detector_weights_path,
            medsam_weights_path,
        )
    ]
    image_path, reason_weights_path, detector_weights_path, medsam_weights_path = paths
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    # The detector and MedSAM2 runners accept their model folder or checkpoint file.
    for path in (detector_weights_path, medsam_weights_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not reason_weights_path.is_dir():
        raise NotADirectoryError(reason_weights_path)
    providers = {
        name: registry.get_model(name)
        for name in (
            "mlx-reason-ct",
            "lung-nodule",
            "medsam2",
        )
    }
    runners = {name: registry.resolve(spec.run) for name, spec in providers.items()}
    # An older Reason-CT provider must fail before creating a partial workflow.
    inspect.signature(runners["mlx-reason-ct"]).bind(
        image_path=image_path,
        weights_path=reason_weights_path,
        output_dir=output_dir,
        tools=TOOLS,
        tool_history=[],
        response_choices=[tool_choice("detect_nodules")],
    )
    stamp = image_path.stat()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    events: list[JsonValue] = []
    record: dict[str, JsonValue] = {
        "schema": "medmlx.reason_ct_tools.v1",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "source": str(image_path),
        "source_size": stamp.st_size,
        "source_mtime_ns": stamp.st_mtime_ns,
        "limits": cast(dict[str, JsonValue], limits.model_dump(mode="json")),
        "hu_window": [float(window[0]), float(window[1])],
        "providers": {
            name: {"run": spec.run, "version": spec.version} for name, spec in providers.items()
        },
        "weights": {
            "mlx-reason-ct": str(reason_weights_path),
            "lung-nodule": str(detector_weights_path),
            "medsam2": str(medsam_weights_path),
        },
        "events": events,
        "requires_human_review": True,
        "clinical_validation": False,
        "action_decoding": "LLGuidance constrained native tool calls; greedy FP32 Metal",
    }
    journal = output_dir / "run.json"
    _write_record(journal, record)
    try:
        scan = prepare_scan(
            image_path, output_dir / "working_ct.nii.gz", assume_mm=limits.assume_mm
        )
        assert scan.original.affine is not None and scan.working.affine is not None
        record["geometry"] = {
            "source_shape": list(scan.original.shape),
            "source_affine": scan.original.affine.tolist(),
            "source_units": scan.original.header.get_xyzt_units()[0],
            "working_shape": list(scan.working.shape),
            "working_affine_mm": scan.working.affine.tolist(),
            "resampled_oblique_or_sheared": scan.resampled,
            "assumed_mm": scan.assumed_mm,
            "mask_restoration": (
                "nearest_neighbour_to_source_grid"
                if scan.resampled
                else "inverse_orientation_to_source_grid"
            ),
        }
        detected = False
        candidates: list[Candidate] = []
        measurements: list[Measurement] = []
        history: list[dict[str, str]] = []
        request = (
            f"{limits.prompt}\nUse the provided tools on this CT. First call detect_nodules. "
            f"Segment at least one returned candidate if available, at most {limits.max_segments}. "
            "Use only returned candidate IDs. Call finish after the requested analysis. "
            "Scores are detector scores, not cancer probabilities. "
            "Tool results are measurements, not diagnoses. Do not invent masks or measurements. "
            "If no candidates pass the configured cutoff, report that limited result."
        )
        for step in range(limits.max_steps):
            if not detected:
                choices = [tool_choice("detect_nodules")]
            else:
                done = {m.candidate_id for m in measurements}
                choices = [
                    tool_choice("segment_candidate", c.candidate_id)
                    for c in candidates
                    if c.candidate_id not in done and len(done) < limits.max_segments
                ]
                if measurements or not candidates:
                    choices.append(tool_choice("finish"))
            step_dir = output_dir / f"reason-{step + 1:02d}"
            events.append(
                {"step": step + 1, "event": "reason_requested", "allowed_calls": list(choices)}
            )
            _write_record(journal, record)
            runners["mlx-reason-ct"](
                image_path=scan.working_path,
                weights_path=reason_weights_path,
                output_dir=step_dir,
                prompt=request,
                tools=TOOLS,
                tool_history=history,
                response_choices=choices,
                anatomy_region="chest",
                max_new_tokens=limits.max_new_tokens,
            )
            response = (step_dir / "report.txt").read_text().strip()
            history.append({"role": "assistant", "content": response})
            events.append({"step": step + 1, "event": "model_response", "text": response})
            selected: Candidate | None = None
            try:
                action = parse_action(response)
                if isinstance(action, Detect) and detected:
                    raise ValueError("Detection has already completed; use the returned IDs")
                if isinstance(action, Segment):
                    selected = next(
                        (c for c in candidates if c.candidate_id == action.candidate_id), None
                    )
                    if selected is None:
                        raise ValueError("Unknown candidate ID; detect first and use a returned ID")
                    if any(m.candidate_id == action.candidate_id for m in measurements):
                        raise ValueError("This candidate has already been segmented")
                    if len(measurements) >= limits.max_segments:
                        raise ValueError("Segmentation limit reached; provide your final answer")
                if isinstance(action, Finish) and (
                    not detected or (candidates and not measurements)
                ):
                    raise ValueError(
                        "Call detection and segment a returned candidate before answering"
                    )
            except ValueError as error:
                feedback: dict[str, JsonValue] = {"error": str(error), "executed": False}
                events.append({"step": step + 1, "event": "request_rejected", **feedback})
                history.append({"role": "tool", "content": json.dumps(feedback)})
                _write_record(journal, record)
                continue
            if isinstance(action, Detect):
                directory = output_dir / "detection"
                runners["lung-nodule"](
                    image_path=scan.working_path,
                    weights_path=detector_weights_path,
                    output_dir=directory,
                    resample_raw_input=True,
                )
                candidates, outside = ground_candidates(
                    scan,
                    directory / "detections.json",
                    threshold=limits.score_threshold,
                    limit=limits.max_candidates,
                    padding_mm=limits.padding_mm,
                )
                detected = True
                feedback = {
                    "tool": "detect_nodules",
                    "score_threshold": limits.score_threshold,
                    "candidates": [cast(dict[str, JsonValue], c.model_dump()) for c in candidates],
                    "outside_image_count": outside,
                    "candidate_limit": limits.max_candidates,
                }
            elif isinstance(action, Segment):
                assert selected is not None
                directory = output_dir / selected.candidate_id
                runners["medsam2"](
                    image_path=scan.working_path,
                    weights_path=medsam_weights_path,
                    output_dir=directory / "medsam2",
                    box_xyxy_pixels=selected.box_xyxy,
                    frame_index=selected.frame_idx,
                    propagation="both",
                    slice_range=selected.slice_range,
                    hu_window=window.tolist(),
                )
                measurement = restore_mask(
                    scan,
                    selected,
                    directory / "medsam2/medsam2_mask.nii.gz",
                    directory / "mask.nii.gz",
                )
                measurements.append(measurement)
                feedback = {
                    "tool": "segment_candidate",
                    "measurement": cast(dict[str, JsonValue], measurement.model_dump()),
                }
            else:
                evidence = {
                    "candidates": [c.model_dump() for c in candidates],
                    "measurements": [m.model_dump(exclude={"mask_path"}) for m in measurements],
                    "score_threshold": limits.score_threshold,
                }
                runners["mlx-reason-ct"](
                    image_path=scan.working_path,
                    weights_path=reason_weights_path,
                    output_dir=output_dir / "interpretation",
                    anatomy_region="chest",
                    max_new_tokens=limits.max_new_tokens,
                    prompt=(
                        f"{limits.prompt}\nThe following tools have now completed on this CT. "
                        "Explain the result briefly using the candidate IDs. Measurements are "
                        "shown separately; do not invent or estimate additional measurements. "
                        "Detection and segmentation do not establish malignancy. "
                        "A missing candidate does not establish absence of disease. "
                        "Tool evidence:\n" + json.dumps(evidence, allow_nan=False)
                    ),
                )
                report = (output_dir / "interpretation/report.txt").read_text().strip()
                now = image_path.stat()
                if (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns) != (
                    now.st_ino,
                    now.st_size,
                    now.st_mtime_ns,
                ):
                    raise ValueError("Source CT changed during the workflow")
                report_path = output_dir / "report.txt"
                report_path.write_text(report + "\n")
                record["measurements"] = [
                    cast(dict[str, JsonValue], m.model_dump()) for m in measurements
                ]
                record["candidates"] = [
                    cast(dict[str, JsonValue], c.model_dump()) for c in candidates
                ]
                viewer = write_review(
                    scan,
                    candidates,
                    measurements,
                    report,
                    output_dir,
                    (float(window[0]), float(window[1])),
                    score_threshold=limits.score_threshold,
                )
                record["status"] = "completed"
                events.append({"step": step + 1, "event": "finished"})
                return {
                    "outputs": {
                        "review": str(viewer),
                        "report": str(report_path),
                        "run": str(journal),
                    },
                    "details": record,
                }
            events.append({"step": step + 1, "event": "tool_result", **feedback})
            history.append({"role": "tool", "content": json.dumps(feedback, allow_nan=False)})
            _write_record(journal, record)
        raise RuntimeError(
            f"Reason-CT did not complete within {limits.max_steps} decisions; see {journal}"
        )
    finally:
        if record["status"] != "completed":
            record["status"] = "incomplete"
        record["finished_at"] = datetime.now(UTC).isoformat()
        _write_record(journal, record)
