# MedMLX MCP server

The MedMLX MCP server lets a local MCP client discover installed medical imaging
models, inspect their inputs, and run them on Apple Silicon. It uses standard
input/output (stdio): the client starts `medmlx serve` as a subprocess and
communicates with it over that process's stdin and stdout.

This is the inference server in the `medmlx` repository.

## Install and connect

You need macOS on Apple Silicon, Python 3.13, Metal, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). From a source
checkout, install MCP and the model family you need. This example installs the
segmentation family:

```sh
git clone https://github.com/MedMLX/medmlx.git
cd medmlx
uv sync --locked --no-dev --extra segment --extra mcp
```

Use `generate`, `vision`, or `all` in place of `segment` to install other
families. Runners with automatic assets download their configured Hub files when
`weights_path` is omitted. Optionally run `medmlx setup --all` before working
offline, or pass your own local folder as `weights_path`. The setup command
reports models and workflows that need manual assets. Readiness never downloads
weights. MedGemma requires local weights and uses a
[managed separate-environment bridge](medvlm-bridge.md).

Check the installed environment before connecting a client:

```sh
.venv/bin/medmlx models
.venv/bin/medmlx runtime
```

`models` lists discovered declarations and reports `discovery_errors` and
`discovery_warnings`; `runtime` reports MLX and Metal availability. Discovery alone does not mean
weights are staged or a particular input is valid.

### Codex

Run the following from the checkout. Choose an existing working directory for
the server; the example creates one in your home directory. The absolute path
to the executable ensures that Codex starts the same environment you installed.

```sh
MEDMLX_WORK_ROOT="$HOME/medmlx-work"
mkdir -p "$MEDMLX_WORK_ROOT"
codex mcp add medmlx -- "$PWD/.venv/bin/medmlx" serve --root "$MEDMLX_WORK_ROOT"
codex mcp list
```

Start a new Codex session and check its MCP server list, or use `/mcp` in the
terminal UI. Then ask it to list MedMLX models and report `runtime_info`. See
the [Codex MCP documentation](https://developers.openai.com/codex/mcp/) for
client configuration and troubleshooting. Registration is a client setting;
installing MedMLX does not register it automatically.

For another stdio client, configure the absolute path to `.venv/bin/medmlx` as
the command and `serve`, `--root`, and the absolute working-directory path as
separate arguments. Add `--restrict-root` to reject path arguments that resolve
outside that directory. Consult that client's configuration instructions. The root
must already exist. If `--root` is omitted, the server uses its startup working
directory, which may differ from the directory where you interact with the client.

## Run a model

The client may expose these tools directly or let an agent call them on your
behalf. A typical sequence is:

1. Call `runtime_info` and `list_models` to inspect the host and available IDs.
2. Call `describe_model` for the selected ID. Its `parameters` come from the
   runner signature, and `instructions` reproduce the provider's docstring.
   Follow the provider's requirements for inputs, weights, units, and output.
3. Call `model_readiness` with one model ID and the intended arguments. Fix any
   known blockers before starting a run. If its `weights.source` is
   `download`, call `prepare_model` and wait for that job with `get_job`.
4. Call `run_model`, retain its `job_id`, and call `get_job` until the state is
   terminal. Inspect the result or error and the provider's output files.
5. Call `summarize_output` and `preview_output` on output masks with that `job_id`
   to report named label volumes and show a slice.

A DICOM folder is converted first with `convert_dicom` (see Inputs and outputs).

For a runnable example that creates synthetic inputs and exercises the actual
MCP session without model weights, see the [FDG smoke walkthrough](mcp-smoke.md).

For example, after staging a compatible converted TotalSegmentator model and a
CT NIfTI image, a client can call the following tools with these JSON arguments.
The paths are examples relative to the server root; replace them with your own.
The [TotalSegmentator provider](https://github.com/MedMLX/totalsegmentator-mlx)
documents its model folders, modes, and input requirements.

`describe_model`:

```json
{"model_id": "totalsegmentator"}
```

`model_readiness`:

```json
{
  "model_ids": ["totalsegmentator"],
  "arguments": {
    "weights_path": "models/total",
    "image_path": "inputs/ct.nii.gz",
    "output_dir": "results/ct-total"
  }
}
```

If appropriate for the task, pass the same model ID and runner arguments to
`run_model`:

```json
{
  "model_id": "totalsegmentator",
  "arguments": {
    "weights_path": "models/total",
    "image_path": "inputs/ct.nii.gz",
    "output_dir": "results/ct-total"
  }
}
```

Use the returned `job_id` with `get_job`. For example, replacing the placeholder
ID with the returned value:

```json
{"job_id": "RETURNED_JOB_ID", "wait_seconds": 30}
```

Repeat while the state is `running`. A job can also finish before the first
poll. TotalSegmentator writes `mask.nii.gz` to the selected output directory;
its default is to refuse to replace an existing mask. The server's job result
contains the provider's output paths and details when the run succeeds.

## Inputs and outputs

DICOM is converted before a run: `list_dicom_series` shows the series in a
folder, and `convert_dicom` writes one image series as NIfTI with its geometry and
rescaled values (Hounsfield units for CT). Non-image objects such as SEG, RTSTRUCT
and SR are refused. DICOM conversion needs SimpleITK, installed with `medmlx[all]`
or `medmlx[dicom]`.

After a run, `summarize_output` reads a NIfTI output. It reports each mask label's
voxel count and volume in mL from the header's voxel size. Pass `job_id` to name
labels from that job's matching `outputs` entry. Paths are compared after resolving
them against the server root. If no entry supplies labels, `model_id` supplies the
model's declared names. Without either source, names stay `null`. `names_from`
reports `"outputs"`, `"declaration"` or `null`.

`preview_output` accepts the same `job_id` and `model_id`. It returns a PNG and
the same structured summary, including `labels` and `names_from`. Named labels
appear in a legend. The PNG also stores `labels` as JSON and `names_from` as text
(`outputs`, `declaration` or `null`). Python's `medmlx.preview_output` still returns
PNG bytes. Python calls with `job_id` also pass the owning `Jobs` session as `jobs`.

Previews show the slice with the most labelled voxels in radiological orientation
(patient right on the image left). Pass `image_path` to overlay the mask on its
scan. Neither tool interprets findings.

## Memory

Models share the Mac's unified memory, so one model runs at a time on the machine
across every MedMLX session: each run holds a lock that the operating system
releases when its worker exits, so a crashed run never blocks the next one. A
second `run_model` from any session is refused with `busy` while another model
runs; `memory_status` shows what is running. The guard also refuses a run with
`insufficient_memory` when its effective peak exceeds available memory.
Weight downloads do not count.
`MEDMLX_MAX_RUNS` raises the limit on machines with room for more.

After a successful run, the worker reads `mlx.core.get_peak_memory()` if MLX
is already loaded and reports it as `measured_peak_memory_gb` on the job. Each
job has its own process, so this is that run's peak. The figure covers MLX
allocations only, not NumPy or preprocessing memory, and is reported in decimal
GB. MedMLX never imports MLX just to measure memory.

The highest measured peak for each model ID and package version is saved on
this Mac in `~/.cache/medmlx/peaks.json`; `MEDMLX_PEAKS` overrides that path.
Updates use a process lock and atomic file replacement. A corrupt store is
treated as empty. Measurements without package-version metadata are not saved.
The guard uses the larger of the model's declared `peak_memory_gb` and the
current version's measured peak, so a measurement from a small scan never lowers
the declared figure. An older version's measurements do not apply after an
upgrade.

`describe_model`, the readiness `memory` row and `memory_status.active_runs`
report `peak_memory_gb` (declared), `measured_peak_memory_gb` (the highest saved
peak for that version), `effective_peak_memory_gb` (used for the fit check),
and `peak_memory_source` (`measured`, `declared`, or `null` when neither exists).
Active run rows preserve the figures used when that run was admitted.

## Readiness and jobs

`model_readiness` reports `ready` and `runtime_ready` as the same boolean: the
provider/import checks and Metal check passed. They do not validate a run.
`run_ready` is `false` for a known blocker. It is `true` when every check
passed: the runtime and Metal, argument binding, input and weight paths, weights
supplied as `weights_path` or already cached, and a readable NIfTI header for
every input file. Otherwise it is `null`: no blocker was found, but something
could not be checked, such as a non-NIfTI input or weights the run would download
first. `weights` reports where the weights come from (`argument`, `cache`,
`download` or `manual`), and `image_checks` lists each NIfTI input's shape. No
check runs the model or proves that weights and images are compatible.

When arguments are supplied, exactly one model ID is required. The check binds
argument names and required parameters to the runner signature and checks
conventionally named input and weight paths for existence. It does not inspect
image contents, geometry, checkpoint compatibility, or output collisions.
Provider-authored descriptions are guidance, not independently validated
constraints. `run_model` checks the signature and Metal before starting a worker;
the provider validates the actual inputs during execution and can still fail.

Tool errors carry a code and a next step, for example
`invalid_arguments: ...` followed by `Next: Call describe_model for the runner's
parameter names.` The codes are `unknown_model`, `unknown_job`,
`invalid_arguments`, `manual_setup`, `busy`, `insufficient_memory`, `metal_unavailable`,
`missing_dependency`, `path_outside_root`, `filesystem_error` and `runtime_error`.

Jobs have states `running`, `completed`, `failed`, or `cancelled`. `get_job`
accepts `wait_seconds` from 0 to 60; its default is 0. It returns a log tail and,
once available, a result or error. `list_jobs` lists this session's jobs, and
`cancel_job` stops a running worker and its child processes. Jobs are owned by
the server session: disconnecting cancels active workers and removes their
temporary control files. A later session cannot retrieve those job IDs. Do not
assume cancellation removes output files a provider has already written.

A successful result with invalid `outputs` metadata carries `outputs_error` on
the job. Its result stays unchanged and its state remains `completed`. Output
tools ignore that list and use an explicit `model_id` for names when given.

## Tool reference

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `runtime_info` | None | Report server root, platform, package versions, and MLX/Metal status. |
| `list_packages` | None | Show model families, runtime packages, installed versions, and discovery errors. |
| `list_models` | Optional `task`, `family` | List discovered models; filters match declared task or `segment`, `generate`, or `vision` family. |
| `list_checkpoints` | Optional `model_id` | List cataloged source references and channel order; this neither downloads nor qualifies assets. |
| `describe_model` | `model_id` | Show one runner's parameters, provider-authored instructions, declared `labels` and `example_arguments`, and declared, measured and effective peak memory. |
| `model_readiness` | Optional `model_ids`, `arguments` | Check runtime readiness and, for one model, argument binding, input and weight paths, weights and NIfTI headers. |
| `prepare_model` | `model_id` | Download a model's pinned weights as a job; the result gives the `weights_path` folder. |
| `run_model` | `model_id`, `arguments` | Start a supervised worker and return its job record. |
| `get_job` | `job_id`, optional `wait_seconds` | Get a job's state, log tail, and eventual result or error. |
| `list_jobs` | None | List jobs in the current server session. |
| `cancel_job` | `job_id` | Stop a running job and its child processes. |
| `memory_status` | None | Report total and available memory and every MedMLX model run holding memory on this Mac, including the declared, measured and effective peaks used at admission. |
| `list_dicom_series` | `directory` | List a folder's DICOM series with modality, description and file count. |
| `convert_dicom` | `directory`, `output_path`, optional `series_uid` | Write one DICOM image series as NIfTI for model input; never replaces a file. |
| `summarize_output` | `path`, optional `job_id`, `model_id` | Report geometry, label counts and volumes. Names come from matching run outputs, then the explicit model declaration. Report `names_from`. |
| `preview_output` | `path`, optional `image_path`, `job_id`, `model_id` | Return a PNG with a label legend and the same structured summary as `summarize_output`. |

If models are missing, check the installed family and `discovery_errors`. If a
run is blocked, inspect `runtime_info`, the readiness reason and setup hint, and
the job's error and log tail. `list_checkpoints` is a source catalog, not a list
of installed or validated model weights.

## Files and permissions

`--root` sets the workers' working directory for relative paths; by itself it is
not a filesystem sandbox. With `--restrict-root`, readiness checks and runs reject
path arguments, outputs included, that resolve outside the root; runners can
still read their own managed weight caches. Absolute paths and paths outside the root remain subject to
the operating system's and client's permissions. Model runners can read inputs,
write outputs, and, depending on the provider and arguments, access the network.
Results and log tails are returned to the MCP client and may contain file paths
or provider output. Configure client access and data handling accordingly.

MedMLX is research software. Numerical parity does not establish clinical
equivalence, and model outputs require human review.
