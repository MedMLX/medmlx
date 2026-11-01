#!/usr/bin/env bash
# Canonical macOS/Linux setup entrypoint. Target folders contain environment.yml
# specs only; Python project dependencies come from pyproject.toml + uv.lock.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TARGET_OVERRIDE=""
HOST_CHECK=0
PRINT_TARGET=0
RUN_COMMAND=()

usage() {
  cat <<'EOF'
Usage: bash environment/setup.sh [OPTIONS]

Options:
  --target macos|linux|linux-dgx  Override target selection.
  --host-check                    Print host/target details without mutation.
  --print-target                  Print the effective target without mutation.
  --run COMMAND [ARG ...]         Run a command in the existing environment.
  -h, --help                      Show this help.
EOF
}

die() {
  printf '[setup] ERROR: %s\n' "$1" >&2
  exit 1
}

detect_target() {
  case "$(uname -s)" in
    Darwin) printf 'macos\n' ;;
    Linux)
      if [[ "$(uname -m)" == "aarch64" && -f /etc/dgx-release ]]; then
        printf 'linux-dgx\n'
      else
        printf 'linux\n'
      fi
      ;;
    *) printf 'unsupported\n' ;;
  esac
}

require_target_host_match() {
  case "$1:$(uname -s)" in
    macos:Darwin | linux:Linux | linux-dgx:Linux) ;;
    *) die "Target '$1' is incompatible with host $(uname -s) $(uname -m)." ;;
  esac
  if [[ "$1" == "linux-dgx" ]]; then
    [[ "$(uname -m)" == "aarch64" ]] || die "linux-dgx requires aarch64."
    command -v nvidia-smi >/dev/null 2>&1 || die "linux-dgx requires nvidia-smi."
  fi
}

validate_repo_contract() {
  local pyproject="${REPO_ROOT}/pyproject.toml"
  local lockfile="${REPO_ROOT}/uv.lock"
  local lock_count=0
  local lock target_requirements

  if [[ -f "$pyproject" && ! -f "$lockfile" ]] || \
    [[ ! -f "$pyproject" && -f "$lockfile" ]]; then
    die "pyproject.toml and uv.lock must either both exist or both be absent."
  fi
  if [[ -f "$pyproject" ]]; then
    command -v uv >/dev/null 2>&1 ||
      die "Install uv before syncing this Python project."
    (
      cd -- "$REPO_ROOT"
      uv lock --check
    )
  fi
  for target_requirements in "${SCRIPT_DIR}"/*/requirements.txt; do
    [[ ! -f "$target_requirements" ]] ||
      die "Target requirements.txt is unsupported: ${target_requirements}. Declare Python dependencies in pyproject.toml and uv.lock."
  done

  for lock in pnpm-lock.yaml package-lock.json yarn.lock; do
    [[ -f "${REPO_ROOT}/${lock}" ]] && ((lock_count += 1))
  done
  if [[ ! -f "${REPO_ROOT}/package.json" && "$lock_count" -gt 0 ]]; then
    die "A Node lockfile exists without package.json."
  fi

  if [[ -f "${REPO_ROOT}/package.json" ]]; then
    [[ "$lock_count" -eq 1 ]] ||
      die "package.json requires exactly one lockfile (pnpm-lock.yaml, package-lock.json, or yarn.lock); found ${lock_count}."
    [[ -f "${REPO_ROOT}/devtools/node-package.mjs" ]] ||
      die "package.json requires devtools/node-package.mjs."
    command -v node >/dev/null 2>&1 ||
      die "Node is required on the host to validate package.json before setup."
    (
      cd -- "$REPO_ROOT"
      node devtools/node-package.mjs validate
    )
  fi
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --target)
        [[ "$#" -ge 2 && "$2" != -* ]] || die "--target requires a value."
        [[ -z "$TARGET_OVERRIDE" ]] || die "--target may be provided only once."
        TARGET_OVERRIDE="$2"
        shift 2
        ;;
      --target=*)
        [[ -z "$TARGET_OVERRIDE" ]] || die "--target may be provided only once."
        TARGET_OVERRIDE="${1#--target=}"
        [[ -n "$TARGET_OVERRIDE" ]] || die "--target requires a value."
        shift
        ;;
      --host-check) HOST_CHECK=1; shift ;;
      --print-target) PRINT_TARGET=1; shift ;;
      --run)
        shift
        [[ "$#" -gt 0 ]] || die "--run requires a command."
        RUN_COMMAND=("$@")
        break
        ;;
      -h | --help) usage; exit 0 ;;
      *) usage >&2; die "Unknown argument: $1" ;;
    esac
  done
}

select_mamba() {
  if command -v mamba >/dev/null 2>&1; then
    printf 'mamba\n'
  elif command -v conda >/dev/null 2>&1; then
    printf 'conda\n'
  else
    die "Install mamba or conda before running setup."
  fi
}

read_env_name() {
  local line_count line name
  line_count="$(grep -Ec '^[[:space:]]*name[[:space:]]*:' "$1" || true)"
  [[ "$line_count" -eq 1 ]] ||
    die "Expected exactly one canonical 'name:' entry in $1; found $line_count."
  line="$(grep -E '^[[:space:]]*name[[:space:]]*:' "$1")"
  if [[ "$line" =~ ^[[:space:]]*name[[:space:]]*:[[:space:]]*([A-Za-z0-9][A-Za-z0-9._-]*)[[:space:]]*$ ]]; then
    name="${BASH_REMATCH[1]}"
  else
    die "Environment name must be an unquoted [A-Za-z0-9][A-Za-z0-9._-]* token in $1."
  fi
  [[ "$name" != "base" ]] || die "Environment name 'base' is not allowed."
  printf '%s\n' "$name"
}

create_or_update_env() {
  if environment_exists; then
    CONDA_ALWAYS_YES=true "$MAMBA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE"
  else
    "$MAMBA_BIN" env create -n "$ENV_NAME" -f "$ENV_FILE" -y
  fi
}

environment_exists() {
  "$MAMBA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

run_in_env() {
  environment_exists ||
    die "Environment '$ENV_NAME' does not exist; run 'make env' first."
  if [[ "$MAMBA_BIN" == "conda" ]]; then
    exec "$MAMBA_BIN" run -n "$ENV_NAME" --no-capture-output -- "${RUN_COMMAND[@]}"
  fi
  exec "$MAMBA_BIN" run -n "$ENV_NAME" "${RUN_COMMAND[@]}"
}

run_env_capture() {
  if [[ "$MAMBA_BIN" == "conda" ]]; then
    "$MAMBA_BIN" run -n "$ENV_NAME" -- "$@"
  else
    "$MAMBA_BIN" run -n "$ENV_NAME" "$@"
  fi
}

run_env_live() {
  if [[ "$MAMBA_BIN" == "conda" ]]; then
    "$MAMBA_BIN" run -n "$ENV_NAME" --no-capture-output -- "$@"
  else
    "$MAMBA_BIN" run -n "$ENV_NAME" "$@"
  fi
}

sync_python_project() {
  [[ -f "${REPO_ROOT}/pyproject.toml" ]] || return 0

  local env_prefix
  env_prefix="$(run_env_capture python -c 'import sys; print(sys.prefix)')"
  [[ -n "$env_prefix" && -d "$env_prefix" ]] || die "Could not resolve environment prefix for '$ENV_NAME'."
  (
    cd -- "$REPO_ROOT"
    UV_PROJECT_ENVIRONMENT="$env_prefix" uv sync --locked --inexact
  )
}

install_node_project() {
  [[ -f "${REPO_ROOT}/package.json" ]] || return 0
  (
    cd -- "$REPO_ROOT"
    run_env_live node devtools/node-package.mjs install
  )
}

parse_args "$@"
if [[ "$HOST_CHECK" -eq 1 && "$PRINT_TARGET" -eq 1 ]] || \
  [[ "${#RUN_COMMAND[@]}" -gt 0 && "$((HOST_CHECK + PRINT_TARGET))" -gt 0 ]]; then
  die "Choose only one of --host-check, --print-target, or --run."
fi
DETECTED_TARGET="$(detect_target)"
TARGET_OS="${TARGET_OVERRIDE:-$DETECTED_TARGET}"
case "$TARGET_OS" in
  macos | linux | linux-dgx) ;;
  *) die "Unsupported target '$TARGET_OS'. Expected macos, linux, or linux-dgx." ;;
esac

require_target_host_match "$TARGET_OS"

if [[ "$PRINT_TARGET" -eq 1 ]]; then
  printf '%s\n' "$TARGET_OS"
  exit 0
fi
if [[ "$HOST_CHECK" -eq 1 ]]; then
  printf 'host_os=%s\nhost_arch=%s\ndetected_target=%s\neffective_target=%s\nnvidia_smi=%s\n' \
    "$(uname -s)" "$(uname -m)" "$DETECTED_TARGET" "$TARGET_OS" \
    "$(command -v nvidia-smi >/dev/null 2>&1 && printf present || printf absent)"
  exit 0
fi

ENV_FILE="${SCRIPT_DIR}/${TARGET_OS}/environment.yml"
[[ -f "$ENV_FILE" ]] || die "Missing environment spec: $ENV_FILE"
validate_repo_contract
MAMBA_BIN="$(select_mamba)"
ENV_NAME="$(read_env_name "$ENV_FILE")"

if [[ "${#RUN_COMMAND[@]}" -gt 0 ]]; then
  run_in_env
fi

printf '[setup] target=%s env=%s\n' "$TARGET_OS" "$ENV_NAME"
create_or_update_env
sync_python_project
install_node_project
printf '[setup] ready: target=%s env=%s\n' "$TARGET_OS" "$ENV_NAME"
