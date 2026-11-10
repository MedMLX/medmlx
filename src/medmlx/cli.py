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


def inline_arguments(model_id: str, pairs: list[str]) -> dict[str, JsonValue]:
    """Parse ``--arg`` pairs into one runner's keyword arguments.

    ``KEY=VALUE`` converts ``VALUE`` to the type ``KEY`` declares (see
    ``api.text_arguments``); ``KEY:=JSON`` passes a finite JSON value unchanged,
    for lists, mappings, ``null`` and ambiguous unions.

    Raises:
        LookupError: ``model_id`` is not a discovered model.
        ValueError: A pair is malformed or repeated, a name is not a declared
            argument, a text value does not convert, or JSON is invalid or
            non-finite.
    """
    text: dict[str, str] = {}
    literal: dict[str, JsonValue] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        is_json = name.endswith(":")
        name = name.removesuffix(":")
        if not separator or not name.isidentifier():
            raise ValueError(f"Expected KEY=VALUE or KEY:=JSON, got {pair!r}")
        if name in text or name in literal:
            raise ValueError(f"Argument {name!r} is given more than once")
        if not is_json:
            text[name] = value
            continue
        try:
            literal[name] = json.loads(value, parse_constant=_reject_constant)
            json.dumps(literal[name], allow_nan=False)
        except ValueError as error:
            raise ValueError(f"Argument {name!r} is not finite JSON: {error}") from None
    return {**api.text_arguments(model_id, text), **literal}


def _arguments(
    model_id: str, path: Path | None, pairs: list[str] | None,
) -> dict[str, JsonValue] | None:
    """Merge an ``--arguments`` file with ``--arg`` pairs; pairs replace file entries."""
    if path is None and not pairs:
        return None
    return {
        **({} if path is None else read_arguments(path)),
        **inline_arguments(model_id, pairs or []),
    }


def _add_argument_options(command: argparse.ArgumentParser, help_file: str) -> None:
    command.add_argument(
        "--arg", dest="pairs", action="extend", nargs="+", metavar="KEY=VALUE",
        help="Runner argument; values take the declared type. Use KEY:=JSON for lists, "
             "objects and null. See medmlx describe MODEL for the names.",
    )
    command.add_argument("--arguments", type=Path, help=help_file)
    command.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Resolve relative paths against this directory (default: current directory).",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medmlx", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="List installed model declarations.")
    models.add_argument("--task")
    models.add_argument("--family", choices=("segment", "generate", "vision"))
    checkpoints = commands.add_parser("checkpoints", help="List checkpoint sources; no downloads.")
    checkpoints.add_argument("--model-id")
    commands.add_parser("packages", help="Show optional runtime families and versions.")
    commands.add_parser("runtime", help="Probe the MLX Metal runtime.")
    describe = commands.add_parser("describe", help="Show a runner's keyword arguments.")
    describe.add_argument("model_id")
    ready = commands.add_parser("readiness", help="Check model and Metal readiness.")
    ready.add_argument("model_ids", nargs="*")
    _add_argument_options(ready, "Check one model's argument JSON file and paths.")
    setup = commands.add_parser(
        "setup", help="Optionally pre-download model weights for offline use.",
    )
    setup.add_argument(
        "model_id",
        nargs="?",
        help="Installed model ID. Omit to list models; normal runs download automatically.",
    )
    setup.add_argument("--all", action="store_true", help="Pre-download every installed model.")
    setup.add_argument("--refresh", action="store_true", help="Update or repair cached weights.")
    setup.add_argument(
        "--plan", action="store_true", help="Describe setup without changing anything.",
    )
    setup.add_argument("--progress", action="store_true", help="Stream JSON progress on stderr.")
    run = commands.add_parser("run", help="Run a model and wait for its result.")
    run.add_argument("model_id")
    _add_argument_options(run, "JSON keyword-argument file; --arg entries override it.")
    serve = commands.add_parser("serve", help="Serve MCP on stdio (requires the mcp extra).")
    serve.add_argument("--root", type=Path, default=Path.cwd())
    serve.add_argument(
        "--restrict-root", action="store_true",
        help="Reject path arguments, outputs included, that resolve outside the root.",
    )
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


def _serve(root: Path, *, restrict: bool = False) -> int:
    """Serve MCP on stdio until the client disconnects; ``root`` is the workers' cwd.

    ``restrict`` rejects path arguments that resolve outside ``root``.

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

    build_server(root, restrict_paths=restrict).run("stdio")
    return 0


def _execute(options: argparse.Namespace) -> tuple[object, int]:
    """Dispatch one parsed command and return its JSON payload and exit code.

    ``models`` still returns its payload when some entry points failed to load,
    but with exit code 1 so scripts notice the ``discovery_errors``.
    """
    match options.command:
        case "setup":
            if options.all and options.model_id is not None:
                raise ValueError("Choose one model ID or --all")
            if options.refresh and options.model_id is None and not options.all:
                raise ValueError("Use --refresh with a model ID or --all")
            if options.plan or (options.model_id is None and not options.all):
                if options.model_id is None:
                    return api.list_setup_plans(), 0
                return api.setup_plan(options.model_id), 0
            def progress(message: str) -> None:
                if options.progress:
                    print(json.dumps({"message": message}), file=sys.stderr, flush=True)
            def download_progress(event: api.DownloadProgress) -> None:
                print(json.dumps(event), file=sys.stderr, flush=True)
            try:
                if options.all:
                    result = api.setup_all(
                        progress=progress, refresh=options.refresh,
                        download_progress=download_progress if options.progress else None,
                    )
                    return result, 0 if result["complete"] else 1
                assert options.model_id is not None
                return api.setup_model(
                    options.model_id, progress=progress, refresh=options.refresh,
                    download_progress=download_progress if options.progress else None,
                )["arguments"], 0
            except Exception as error:
                # The setup CLI is a reporting boundary for optional provider/Hub errors.
                return {"error": str(error)}, 1
        case "models":
            payload = api.list_models(options.task, options.family)
            return payload, int(bool(payload["discovery_errors"]))
        case "checkpoints":
            return api.list_checkpoints(options.model_id), 0
        case "packages":
            return api.list_packages(), 0
        case "runtime":
            return api.runtime_info(), 0
        case "describe":
            return api.describe_model(options.model_id), 0
        case "readiness":
            arguments = None
            if options.arguments is not None or options.pairs:
                if len(options.model_ids) != 1:
                    raise ValueError("Supply exactly one model_id when checking arguments")
                arguments = _arguments(options.model_ids[0], options.arguments, options.pairs)
            return api.model_readiness(
                options.model_ids or None, arguments=arguments, root=options.root,
            ), 0
        case "run":
            arguments = _arguments(options.model_id, options.arguments, options.pairs)
            return api.run(options.model_id, arguments=arguments or {}, root=options.root), 0
        case "serve":
            return None, _serve(options.root, restrict=options.restrict_root)
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
