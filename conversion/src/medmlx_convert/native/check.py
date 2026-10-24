"""Gate saved MLX output and exact captured-program replay against the original model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from medmlx_core.graph.runtime import Session

from medmlx_convert.boundary import as_array, as_tensors, function
from medmlx_convert.evidence import OutputComparison, Verdict, compare_output
from medmlx_convert.native.capture import convert
from medmlx_convert.provenance import BuildSource


@dataclass(frozen=True)
class NativeEvidence:
    """One-input-set engineering verdict; no image pipeline or scientific qualification."""

    source: BuildSource
    comparisons: tuple[OutputComparison, ...]
    artifacts: tuple[str, ...]
    operators: tuple[str, ...]
    mlx_version: str
    verdict: Verdict
    schema_version: int = 3
    scope: str = ("Fixed-shape CPU FP32 source versus saved MLX Metal network; "
                  "host pipeline not tested")


def convert_and_check(
    model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], folder: Path, *,
    source: BuildSource, tolerance: float,
) -> NativeEvidence:
    """Convert, reload and compare both stages directly to untouched source output.

    Persist source inputs/outputs, stage comparisons and artifact filenames.
    A failed numerical comparison persists a FAIL receipt then raises. A
    conversion/runtime exception raises without creating a passing receipt.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Native tolerance must be finite and nonnegative")
    converted = convert(model, inputs, folder)
    with torch.no_grad():
        original = model(*inputs)
    references = (original,) if isinstance(original, torch.Tensor) else as_tensors(original)
    native = Session.load(folder)(*(mx.array(value.detach().numpy()) for value in inputs))
    comparisons: list[OutputComparison] = []
    stages = (("torch_capture", converted.captured_outputs),
              ("torch_functional", converted.functional_outputs), ("mlx_metal", native))
    for stage, outputs in stages:
        if len(outputs) != len(references):
            raise ValueError(f"{stage}: output count differs from original source")
        for index, (actual, expected) in enumerate(zip(outputs, references, strict=True)):
            host = (actual.detach().numpy() if isinstance(actual, torch.Tensor)
                    else as_array(function(np, "asarray")(actual)))
            comparisons.append(compare_output(stage, f"output_{index}", as_array(host),
                                               expected.detach().numpy(), tolerance))
    fixtures = {f"input_{i}": value.detach().numpy() for i, value in enumerate(inputs)}
    fixtures.update({f"output_{i}": value.detach().numpy() for i, value in enumerate(references)})
    function(np, "savez")(folder / "source.npz", **fixtures)
    assets = ("graph.json", "weights.safetensors", "source.npz")
    verdict = Verdict.PASS if all(c.verdict == Verdict.PASS for c in comparisons) else Verdict.FAIL
    evidence = NativeEvidence(source, tuple(comparisons), assets,
                              tuple(sorted({str(call.op) for call in converted.graph.calls})),
                              version("mlx"), verdict)
    (folder / "verification.json").write_text(
        json.dumps(asdict(evidence), indent=2, allow_nan=False) + "\n")
    if verdict == Verdict.FAIL:
        raise ValueError(f"Saved native MLX conversion failed parity: {folder}/verification.json")
    return evidence
