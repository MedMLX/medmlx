"""Execute native Qwen-template tool requests under a bounded CT workflow policy."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

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


_CANDIDATE_ID = re.compile(r"candidate-[0-9]{3,}")


def read_action(details: Mapping[str, object]) -> Action:
    """Admit the one tool call of a Reason-CT answer as a workflow action.

    Reason-CT parses its native call markup and checks each call against ``TOOLS``:
    ``details["tool_calls"]`` lists them as names and arguments, and
    ``details["tool_call_error"]`` is set when the markup is malformed, names an
    undeclared tool or parameter, or omits a required one. Exactly one call is
    admitted, and a candidate ID only in the form the workflow assigns. Final
    interpretation is generated separately after the finish action.

    Raises:
        ValueError: The answer holds no usable call, several calls, or a
            candidate ID of another form.
    """
    error = details.get("tool_call_error")
    if error is not None:
        raise ValueError(f"Unusable tool call: {error}")
    calls = details.get("tool_calls")
    if not isinstance(calls, list) or len(cast(list[object], calls)) != 1:
        raise ValueError("Return one complete native tool call, with no suffix or extra calls")
    call = cast(Mapping[str, object], calls[0])
    name = call.get("name")
    arguments = cast(Mapping[str, object], call.get("arguments") or {})
    if name == "detect_nodules" and not arguments:
        return Detect(action="detect_nodules")
    if name == "finish" and not arguments:
        return Finish(action="finish")
    candidate_id = arguments.get("candidate_id")
    if (
        name == "segment_candidate"
        and isinstance(candidate_id, str)
        and _CANDIDATE_ID.fullmatch(candidate_id)
    ):
        return Segment(action="segment_candidate", candidate_id=candidate_id)
    raise ValueError("Unsupported tool or arguments; use detection, a candidate ID, or finish")


def _answer(result: object) -> tuple[str, Mapping[str, object]]:
    """Return the answer text and ``details`` of one Reason-CT session result."""
    details = cast(Mapping[str, Mapping[str, object]], result)["details"]
    report = details["report"]
    if not isinstance(report, str):
        raise RuntimeError("Reason-CT returned no answer text")
    return report.strip(), details


class _ReasonSession(Protocol):
    """The Reason-CT ``load_model`` session: reports about one CT, then ``close``."""

    def run(self, **arguments: object) -> object: ...

    def close(self) -> None: ...


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
    # Every provider takes its published model folder.
    for path in (reason_weights_path, detector_weights_path, medsam_weights_path):
        if not path.is_dir():
            raise NotADirectoryError(path)
    providers = {
        name: registry.get_model(name)
        for name in (
            "mlx-reason-ct",
            "lung-nodule",
            "medsam2",
        )
    }
    runners = {
        name: registry.resolve(providers[name].run) for name in ("lung-nodule", "medsam2")
    }
    # Every decision is about the same working CT, so one Reason-CT session per run
    # loads the model and encodes the CT once; the one-shot operation repeats both
    # at each step. MedMLX runtimes define load_model next to their operation. A
    # provider without it, or with an older signature, fails here, before a partial
    # workflow is created.
    load_reason = registry.resolve(
        providers["mlx-reason-ct"].run.partition(":")[0] + ":load_model"
    )
    # Every decision is a constrained choice among a few calls. The fast mode keeps
    # the decoder state up to the last 64 tokens before the question, so a decision
    # decodes only the conversation; its logits differ from a whole prefill at
    # rounding level, which changed none of 248 constrained answers in Reason-CT's
    # own comparison. The wording of the free-text interpretation can differ from a
    # one-shot report at a near-tie token.
    session_options: dict[str, object] = {
        "image_path": image_path,
        "weights_path": reason_weights_path,
        "anatomy_region": "chest",
        "prefill_reuse": "fast",
    }
    inspect.signature(load_reason).bind(**session_options)
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
    reason: _ReasonSession | None = None
    try:
        scan = prepare_scan(
            image_path, output_dir / "working_ct.nii.gz", assume_mm=limits.assume_mm
        )
        reason = cast(
            _ReasonSession, load_reason(**session_options | {"image_path": scan.working_path})
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
            response, details = _answer(
                reason.run(
                    output_dir=step_dir,
                    prompt=request,
                    tools=TOOLS,
                    tool_history=history,
                    response_choices=choices,
                    max_new_tokens=limits.max_new_tokens,
                )
            )
            history.append({"role": "assistant", "content": response})
            events.append({"step": step + 1, "event": "model_response", "text": response})
            selected: Candidate | None = None
            try:
                action = read_action(details)
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
                    directory / "medsam2/mask.nii.gz",
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
                interpretation = reason.run(
                    output_dir=output_dir / "interpretation",
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
                report, _ = _answer(interpretation)
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
        if reason is not None:
            reason.close()
        if record["status"] != "completed":
            record["status"] = "incomplete"
        record["finished_at"] = datetime.now(UTC).isoformat()
        _write_record(journal, record)
