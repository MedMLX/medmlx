"""The unified MedMLX command for discovery, execution, MCP and conversion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn, cast

from medmlx import api
from medmlx.payloads import JsonValue


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Run arguments require finite JSON numbers, found {value}")


def read_arguments(path: Path) -> dict[str, JsonValue]:
    """Read a finite JSON object containing one model runner's keyword arguments.

    Raises:
        OSError: The file cannot be read.
        ValueError: The text is not JSON, is not an object, or contains ``NaN``,
            ``Infinity`` or a number too large for a finite float.
    """
    value: object = json.loads(path.read_text(), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Run arguments must be a JSON object")
    # JSON exponents such as 1e400 can overflow without invoking parse_constant.
    json.dumps(value, allow_nan=False)
    return cast(dict[str, JsonValue], value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medmlx", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="List installed model declarations.")
    models.add_argument("--task")
    models.add_argument("--family", choices=("segment", "generate", "vision"))
    commands.add_parser("packages", help="Show optional runtime families and versions.")
    commands.add_parser("runtime", help="Probe the MLX Metal runtime.")
    describe = commands.add_parser("describe", help="Show a runner's keyword arguments.")
    describe.add_argument("model_id")
    ready = commands.add_parser("readiness", help="Check model and Metal readiness.")
    ready.add_argument("model_ids", nargs="*")
    run = commands.add_parser("run", help="Run a model and wait for its result.")
    run.add_argument("model_id")
    run.add_argument("--arguments", type=Path, required=True, help="JSON keyword-argument file.")
    run.add_argument("--root", type=Path, default=Path.cwd())
    serve = commands.add_parser("serve", help="Serve MCP on stdio (requires the mcp extra).")
    serve.add_argument("--root", type=Path, default=Path.cwd())
    convert = commands.add_parser("convert", help="Run a source-checkout conversion recipe.")
    convert.add_argument("recipe")
    convert.add_argument("recipe_arguments", nargs=argparse.REMAINDER)
    return parser


def _convert(recipe: str, arguments: list[str]) -> int:
    """Run ``conversion/recipes/<recipe>/export.py`` and return its exit code.

    The recipe runs through ``conversion/environment/setup.sh --run`` with
    ``conversion/`` as the working directory, in the conda environment that
    ``make env-convert`` created. Nothing is installed or synced here, and a
    missing environment makes the script fail. A leading ``--`` in
    ``arguments`` is dropped before the rest is passed to the recipe.

    Raises:
        FileNotFoundError: This is not a source checkout containing ``conversion/``.
        ValueError: ``recipe`` is not a directory under ``conversion/recipes``.
    """
    conversion = Path(__file__).resolve().parents[2] / "conversion"
    if not (conversion / "environment/setup.sh").is_file():
        raise FileNotFoundError(
            "Conversion requires the MedMLX source checkout and make env-convert"
        )
    recipes = {path.parent.name: path for path in (conversion / "recipes").glob("*/export.py")}
    if recipe not in recipes:
        raise ValueError(
            f"Unknown conversion recipe {recipe!r}; choose: {', '.join(sorted(recipes))}"
        )
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return subprocess.run(
        [
            "bash",
            str(conversion / "environment/setup.sh"),
            "--run",
            "python",
            str(recipes[recipe]),
            *arguments,
        ],
        cwd=conversion,
        check=False,
    ).returncode


def _serve(root: Path) -> int:
    """Serve MCP on stdio until the client disconnects; ``root`` is the workers' cwd.

    Raises:
        NotADirectoryError: ``root`` is not an existing directory.
        ImportError: The optional ``mcp`` package is not installed.
    """
    if not root.is_dir():
        raise NotADirectoryError(root)
    try:
        version("mcp")
    except PackageNotFoundError as error:
        raise ImportError(
            "MCP is optional; install it with uv sync --locked --extra mcp"
        ) from error
    from medmlx.server import build_server

    build_server(root).run("stdio")
    return 0


def _execute(options: argparse.Namespace) -> tuple[object, int]:
    """Dispatch one parsed command and return its JSON payload and exit code.

    ``models`` still returns its payload when some entry points failed to load,
    but with exit code 1 so scripts notice the ``discovery_errors``.
    """
    match options.command:
        case "models":
            payload = api.list_models(options.task, options.family)
            return payload, int(bool(payload["discovery_errors"]))
        case "packages":
            return api.list_packages(), 0
        case "runtime":
            return api.runtime_info(), 0
        case "describe":
            return api.describe_model(options.model_id), 0
        case "readiness":
            return api.model_readiness(options.model_ids or None), 0
        case "run":
            return api.run(
                options.model_id, arguments=read_arguments(options.arguments), root=options.root
            ), 0
        case "serve":
            return None, _serve(options.root)
        case "convert":
            return None, _convert(options.recipe, options.recipe_arguments)
        case _:
            raise ValueError(f"Unknown command {options.command!r}")


def main(argv: list[str] | None = None) -> int:
    """Execute one action, printing JSON results and returning nonzero on failure.

    Results are printed to stdout as indented JSON with sorted keys. The exit
    code is 0 on success and 1 when the action raises ``OSError``,
    ``ValueError``, ``LookupError``, ``RuntimeError`` or ``ImportError`` (the
    message goes to stderr as ``medmlx: <message>``). ``convert`` returns the
    recipe's own exit code. Invalid command-line syntax exits with status 2
    through ``argparse``.
    """
    options = _parser().parse_args(argv)
    try:
        payload, code = _execute(options)
    except (OSError, ValueError, LookupError, RuntimeError, ImportError) as error:
        print(f"medmlx: {error}", file=sys.stderr)
        return 1
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
