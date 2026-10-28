"""Compare overlapping Apple-runtime calls with a saved PyTorch fixture.

Run with ``USE_OS_COREAI=1`` and an existing export stem. A passing run does
not rule out an intermittent error; any failed call is reported and exits 1.
Retain every call's metrics and the first failing input/reference/output so a
failure can be investigated after the process exits. Core ML is an optional
control using the same saved fixture and ALL compute units.
"""

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import numpy.typing as npt
from coreai.runtime import AIModel, NDArray, SpecializationOptions

from medmlx_convert.boundary import Array, as_array, attribute, function, item, library


@dataclass(frozen=True)
class CallResult:
    max_abs_difference: float
    label_agreement: float
    seconds: float


async def reproduce(
    stem: Path, rounds: int, concurrency: int, tolerance: float,
    backend: str = "coreai", output_dir: Path | None = None,
) -> bool:
    """Run a shared saved model on identical inputs and compare each completed output."""
    if backend == "coreai" and not SpecializationOptions.is_supported():
        raise RuntimeError("Set USE_OS_COREAI=1 and unset USE_LOCAL_COREAI")
    with np.load(stem.with_suffix(".runtime.npz"), allow_pickle=False) as fixture:
        image = as_array(fixture["input/image"])
        reference = as_array(fixture["output/logits"])
    if image.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError("Expected FP32 input and reference tensors")
    if not np.isfinite(image).all() or not np.isfinite(reference).all():
        raise ValueError("Expected finite input and reference tensors")
    reference_fp32 = cast(npt.NDArray[np.float32], reference)
    reference_labels = reference.argmax(1)
    if output_dir is None:
        root = stem.parent / "overlap"
        root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=f"{stem.name}-{backend}-", dir=root))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"evidence: {output_dir}", flush=True)
    if backend == "coreai":
        model = await AIModel.load(stem.with_suffix(".aimodel"),
                                   specialization_options=SpecializationOptions.default())
        inference = model.load_function("main")

        async def invoke() -> Array:
            outputs = await inference({"image": NDArray(image)})
            result = outputs["logits"]
            if not isinstance(result, NDArray):
                raise TypeError("Expected a Core AI tensor output")
            return as_array(result.numpy()).copy()
    elif backend == "coreml":
        ct = library("coremltools")
        mlmodel = function(attribute(ct, "models"), "MLModel")(
            str(stem.with_suffix(".mlpackage")),
            compute_units=attribute(attribute(ct, "ComputeUnit"), "ALL"),
        )

        async def invoke() -> Array:
            # predict is synchronous; use separate threads so gather really overlaps calls.
            outputs = await asyncio.to_thread(function(mlmodel, "predict"), {"image": image})
            return as_array(item(outputs, "logits")).copy()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    saved_failure = False

    async def call() -> CallResult:
        nonlocal saved_failure
        started = perf_counter()
        output = await invoke()
        seconds = perf_counter() - started
        if output.shape != reference.shape or output.dtype != reference.dtype:
            raise ValueError("Output shape or dtype differs from the fixture")
        if not np.isfinite(output).all():
            np.savez(output_dir / "nonfinite.npz", image=image, reference=reference, output=output)
            raise ValueError("Output contains nonfinite values; tensors saved")
        difference = float(np.max(np.abs(cast(npt.NDArray[np.float32], output) - reference_fp32)))
        if difference > tolerance and not saved_failure:
            np.savez(output_dir / "first-failure.npz",
                     image=image, reference=reference, output=output)
            saved_failure = True
        return CallResult(difference, float((output.argmax(1) == reference_labels).mean()), seconds)

    passed = True
    calls: list[list[CallResult]] = []
    for index in range(rounds):
        results = await asyncio.gather(*(call() for _ in range(concurrency)))
        calls.append(results)
        differences = [result.max_abs_difference for result in results]
        print(f"round {index + 1}: max absolute differences {differences}", flush=True)
        passed &= all(value <= tolerance for value in differences)
        (output_dir / "report.json").write_text(json.dumps({
            "stem": str(stem), "backend": backend, "tolerance": tolerance,
            "concurrency": concurrency, "requested_rounds": rounds, "passed": passed,
            "calls": [[asdict(result) for result in row] for row in calls],
        }, indent=2) + "\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path, help="Export path without its .aimodel suffix")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--backend", choices=("coreai", "coreml"), default="coreai")
    parser.add_argument("--output-dir", type=Path, help="Metrics and first failing tensors")
    args = parser.parse_args()
    if args.rounds < 1 or args.concurrency < 1 or not np.isfinite(args.tolerance):
        parser.error("Rounds and concurrency must be positive and tolerance must be finite")
    if args.tolerance < 0:
        parser.error("Tolerance must be nonnegative")
    if not asyncio.run(reproduce(args.stem, args.rounds, args.concurrency, args.tolerance,
                                args.backend, args.output_dir)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
