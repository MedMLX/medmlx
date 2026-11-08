# MedMLX MCP server

Run medical imaging models on Apple Silicon from an MCP client. The client
starts `medmlx serve` and communicates over standard input/output (stdio).

This is the inference server in the `medmlx` Python package. This guide on `main`
also covers unreleased changes; see the [release notes](release-notes.md#unreleased).

## Install and connect

You need macOS on Apple Silicon, Python 3.13, Metal, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). If uv is missing,
install it with `brew install uv`. Install the released Python package with MCP
and all model families:

```sh
uv tool install --python 3.13 "medmlx[all,mcp]"
```

Use `segment`, `generate`, or `vision` in place of `all` for a smaller install.
Find the absolute path to uv's installed `medmlx` executable and check that
environment before connecting a client:

```sh
MEDMLX="$(uv tool dir --bin)/medmlx"
printf '%s\n' "$MEDMLX"
"$MEDMLX" models
"$MEDMLX" runtime
```

`models` lists discovered declarations and reports `discovery_errors` and
`discovery_warnings`; `runtime` reports MLX and Metal availability. Discovery alone does not mean
weights are staged or a particular input is valid.

Runners with automatic assets download their configured Hub files when
`weights_path` is omitted. Optionally run `medmlx setup --all` before working
offline, or pass your own local folder as `weights_path`. The setup command
reports models and workflows that need manual assets. Readiness never downloads
weights. MedGemma requires local weights and uses a
[managed separate-environment bridge](medvlm-bridge.md).

Choose a working directory for inputs and results. The examples below create
`~/medmlx-work`; replace it with your own folder. `--root` sets the workers'
working directory, so relative input and output paths resolve there. It must
already exist. If omitted, the server uses its startup working directory, which
may differ from the directory where you interact with the client. Add
`--restrict-root` to reject path arguments that resolve outside that directory.

### Claude Code

Register the server in your user configuration, using the absolute executable
path from `uv tool dir --bin`:

```sh
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$(uv tool dir --bin)/medmlx" serve --root "$HOME/medmlx-work"
claude mcp list
```

### Codex

The absolute executable path ensures that Codex starts the installed tool
environment:

```sh
mkdir -p "$HOME/medmlx-work"
codex mcp add medmlx -- "$(uv tool dir --bin)/medmlx" serve --root "$HOME/medmlx-work"
codex mcp list
```

Use `/mcp` in the Codex terminal UI to check its MCP server list. See
the [Codex MCP documentation](https://developers.openai.com/codex/mcp/) for
client configuration and troubleshooting.

### JSON-configured clients

For clients that use `mcpServers`, add this entry to the client's configuration.
Replace both placeholders with absolute paths: the installed executable shown
above and an existing working folder. JSON configuration does not expand shell
variables or `~`.

```json
{
  "mcpServers": {
    "medmlx": {
      "command": "/absolute/path/to/medmlx",
      "args": ["serve", "--root", "/absolute/path/to/medmlx-work"]
    }
  }
}
```

Restart your client or start a new session after registration, then ask it to
list the installed MedMLX models and report `runtime_info`. Registration is a
client setting; installing MedMLX does not register it automatically.

## Source checkout (development)

For development and unreleased changes, install from a source checkout. This
example installs MCP and the segmentation family; use `generate`, `vision`, or
`all` in place of `segment` for other families:

```sh
git clone https://github.com/MedMLX/medmlx.git
cd medmlx
uv sync --locked --no-dev --extra segment --extra mcp
.venv/bin/medmlx models
.venv/bin/medmlx runtime
```

Use the checkout's absolute `.venv/bin/medmlx` path in the client configurations
above. For example, run this from the checkout to register it with Codex:

```sh
mkdir -p "$HOME/medmlx-work"
codex mcp add medmlx -- "$PWD/.venv/bin/medmlx" serve --root "$HOME/medmlx-work"
```

Keep the checkout and its environment at that path so the client can start it.

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

By default, one model runs at a time across all MedMLX sessions on the Mac.
A second run returns `busy`; use `memory_status` to see the active run.
`MEDMLX_MAX_RUNS` raises this limit. Downloads can run alongside inference.

Readiness reports `fits: false` when the expected peak exceeds available
memory, but still allows the run. The estimate uses the larger of the model's
declared peak and the highest measured peak for its installed package version.
Measurements cover MLX allocations in decimal GB; allow additional memory for
preprocessing, macOS and other applications.

Each successful worker reports `measured_peak_memory_gb`. Peaks are saved in
`~/.cache/medmlx/peaks.json` (`MEDMLX_PEAKS` overrides the path).
`describe_model`, readiness and `memory_status` expose the declared, measured
and effective peaks, plus `peak_memory_source`.

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

When supplying arguments, select exactly one model ID. `run_model` checks the
signature and Metal before starting a worker; the provider then validates image
contents, geometry, checkpoint compatibility and output paths during execution.

Tool errors carry a code and a next step, for example
`invalid_arguments: ...` followed by `Next: Call describe_model for the runner's
parameter names.` The codes are `unknown_model`, `unknown_job`,
`invalid_arguments`, `manual_setup`, `busy`, `metal_unavailable`,
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

`--root` sets the working directory for relative paths. Add `--restrict-root`
to reject input and output paths outside it; managed weight caches remain
accessible. Runners use the operating system's permissions and may download
assets. Results and logs are returned to the MCP client.
