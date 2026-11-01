#!/usr/bin/env bash
# Usage: verify.sh <sibling-repository-name>
# Prints git status, lists top-level imports of conversion-only frameworks under
# src/ for manual review (it does not fail on them), reports large fixtures, then
# runs pytest (20-minute limit) and ruff. Exits nonzero on a setup error, a pytest
# failure or ruff findings.
set -euo pipefail

die() { printf 'verify: %s\n' "$1" >&2; exit 2; }
[[ "$#" -eq 1 ]] || die 'usage: verify.sh <sibling-repository-name>'
case "$1" in
  ''|.|..|*[!A-Za-z0-9_.-]*) die 'repository must be a single directory name' ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEDMLX_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${MEDMLX_REPO_ROOT:-$(cd -- "$MEDMLX_DIR/.." && pwd)}"
PYTHON_BIN="${MEDMLX_PYTHON:-$MEDMLX_DIR/.venv/bin/python}"
RUFF_BIN="${MEDMLX_RUFF:-$MEDMLX_DIR/.venv/bin/ruff}"
CORE_SRC="${MEDMLX_CORE_SRC:-$MEDMLX_DIR/packages/core/src}"
[[ -x "$PYTHON_BIN" ]] || die "Python is missing: $PYTHON_BIN; set MEDMLX_PYTHON"
[[ -x "$RUFF_BIN" ]] || die "Ruff is missing: $RUFF_BIN; set MEDMLX_RUFF"
[[ -d "$CORE_SRC" ]] || die "shared core source is missing: $CORE_SRC; set MEDMLX_CORE_SRC"
cd -- "$REPO_ROOT/$1" || die "repository does not exist: $REPO_ROOT/$1"
[[ -d src ]] || die 'repository has no src/ directory'

echo "== $1"
git status --short | sed -n '1,30p'
echo '-- runtime imports of radnn/torch/monai in src (review lazy conversion paths):'
if grep -rnE '^[[:space:]]*(import|from) (radnn|torch|monai)([[:space:].]|$)' src; then
  :
else
  status=$?
  [[ "$status" -eq 1 ]] || exit "$status"
  echo '   none'
fi
echo '-- fixture sizes:'
if [[ -d tests/fixtures ]]; then
  du -sh tests/fixtures
  find tests/fixtures -size +1M
else
  echo '   no fixture directory'
fi

echo '-- tests:'
export PYTHONPATH="$PWD/src:$CORE_SRC${EXTRA_PATH:+:$EXTRA_PATH}"
export PYTHONDONTWRITEBYTECODE=1
# Python supplies the same 20-minute limit on macOS and Linux without requiring
# a separately installed GNU timeout command. pipefail preserves pytest status.
"$PYTHON_BIN" - <<'PY' 2>&1 | tail -3
import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        timeout=1200,
        check=False,
    )
except subprocess.TimeoutExpired:
    raise SystemExit("pytest exceeded the 1200-second limit") from None
raise SystemExit(result.returncode)
PY
echo '-- ruff:'
"$RUFF_BIN" check . 2>&1 | tail -1
