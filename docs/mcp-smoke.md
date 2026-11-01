# Synthetic FDG MCP smoke walkthrough

This example starts a real MedMLX stdio server, discovers `fdg-pet-ct`, checks
readiness, starts a job, waits for it, and checks the written NIfTI. It creates
a tiny synthetic CT, labels, and a zero-activity profile in a temporary directory.
It needs no model checkpoint or patient data. The profile is a software fixture,
not a physiological model or a validation of PET synthesis.

On an Apple Silicon Mac with Metal and [uv](https://docs.astral.sh/uv/), run
from the `medmlx` checkout after installing the generation runtime and MCP:

```sh
uv sync --locked --no-dev --extra generate --extra mcp
```

The script uses the checkout's Python and deletes its temporary inputs and output
when it exits. It prints the runtime, readiness state, and completed job summary.

```sh
.venv/bin/python - <<'PY'
import asyncio
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from mcp import Client, StdioServerParameters


async def call(client, name, arguments):
    response = await client.call_tool(name, arguments)
    if response.is_error or response.structured_content is None:
        raise RuntimeError(f"{name}: {response.content}")
    return response.structured_content


async def exercise(root):
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    for name, values in (
        ("ct", np.zeros((4, 5, 6), dtype=np.float32)),
        ("labels", np.zeros((4, 5, 6), dtype=np.float32)),
    ):
        image = nib.Nifti1Image(values, affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, root / f"{name}.nii.gz")
    profile = {
        "tissues": {
            "unclassified_soft_tissue": {
                "suv_mean": 0.0, "suv_std": 0.0, "maisi_labels": []
            }
        },
        "scanner": {
            "scatter_fraction": 0.0, "randoms_fraction": 0.0,
            "fwhm_mm": [0.0, 0.0, 0.0], "counts_per_suv": 100.0
        },
    }
    (root / "profile.json").write_text(json.dumps(profile))
    arguments = {
        "image_path": "ct.nii.gz", "labels_path": "labels.nii.gz",
        "profile_path": "profile.json", "output_dir": "output"
    }
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medmlx", "serve", "--root", str(root)],
        env=dict(os.environ),
    )
    async with Client(server) as client:
        runtime = await call(client, "runtime_info", {})
        print("Runtime:", runtime["mlx"])
        description = await call(client, "describe_model", {"model_id": "fdg-pet-ct"})
        print("Model:", description["id"])
        readiness = await call(client, "model_readiness", {
            "model_ids": ["fdg-pet-ct"], "arguments": arguments
        })
        row = readiness["models"][0]
        print("Readiness:", row["runtime_ready"], row["run_ready"])
        if row["run_ready"] is False:
            raise RuntimeError(f"Readiness blocker: {row['reason']}")
        job = await call(client, "run_model", {
            "model_id": "fdg-pet-ct", "arguments": arguments
        })
        while job["state"] == "running":
            job = await call(client, "get_job", {
                "job_id": job["job_id"], "wait_seconds": 30
            })
        if job["state"] != "completed":
            raise RuntimeError(f"Job {job['state']}: {job.get('error')}\n{job['log_tail']}")
        output = nib.load(job["result"]["outputs"]["pet"])
        assert output.shape == (4, 5, 6)
        assert output.get_data_dtype() == np.dtype(np.float32)
        assert output.header.get_xyzt_units()[0] == "mm"
        np.testing.assert_allclose(output.affine, affine, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(output.get_fdata(), np.zeros((4, 5, 6)))
        print("Completed:", job["state"], "shape:", output.shape, "units: SUV")


with TemporaryDirectory(prefix="medmlx-mcp-smoke-") as directory:
    asyncio.run(exercise(Path(directory).resolve()))
PY
```

`run_ready: null` is expected after a successful preflight: readiness checks
argument binding and conventional input-path existence, not image or profile
contents. The worker performs those input checks when the job runs. A successful
zero-output fixture confirms this wiring and its stated checks; it does not
establish accuracy on real images.
