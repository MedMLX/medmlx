"""Generate a command-line front end from a runner's signature and docstring.

A runtime's ``run`` function declares its keyword parameters once, with types and
an ``Args:`` section. ``main(run, prog="x-mlx")`` turns them into one command;
runtimes that keep extra subcommands (downloads, label listings) call
``add_options`` on their own subparser and ``call`` on the parsed namespace.

Each keyword parameter becomes ``--name`` with underscores as hyphens, required
when it has no default. The annotation selects the parser:

- ``str``, ``Path`` and path-like unions: text (``Path`` annotations receive a
  ``Path``; every other path-like receives ``str``).
- ``int`` and ``float``: numbers.
- ``bool``: ``--name`` and ``--no-name``.
- ``Literal[...]``: the listed choices.
- ``X | None``: the option for ``X``, optional.
- ``Sequence[X]``, ``list[X]``, ``tuple[X, ...]`` of text or numbers, and
  ``X | Sequence[X]``: one option taking several values.
- ``Mapping[str, X]``: repeated ``key=value``.
- Anything else: one JSON value, or ``@file`` to read the JSON from a file.

An option the user omits is not passed, so the runner's own default applies.
Bind a parameter the command should not expose with ``functools.partial``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import NoneType, SimpleNamespace, UnionType
from typing import (
    Literal,
    NoReturn,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

type Runner = Callable[..., object]
type Converter = Callable[[str], object]

_FAILURES = (OSError, ValueError, RuntimeError, ImportError)
_KEYWORDS = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def main(
    run: Runner,
    *,
    prog: str,
    distribution: str | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Run ``run`` as a single-command CLI and return the exit status.

    The parser has one option per keyword parameter of ``run`` (see the module
    docstring), ``--help`` with the runner's docstring, and ``--version`` showing
    the installed version of ``distribution``, which defaults to ``prog``; name it
    when the command is not called after its distribution. The result prints on stdout
    as indented JSON and the status is 0. An ``OSError``, ``ValueError``,
    ``RuntimeError`` or ``ImportError`` from ``run`` prints
    ``prog: ErrorType: message`` on stderr, then ``prog: hint: ...`` when the
    error has a ``hint``, and the status is 1. Usage errors exit with status 2.
    """
    summary = (inspect.getdoc(_function(run)) or "").partition("\n")[0]
    parser = argparse.ArgumentParser(prog=prog, description=summary or None)
    parser.add_argument("--version", action="version", version=version(distribution or prog))
    add_options(parser, run)
    arguments = parser.parse_args(argv)
    return execute(prog, lambda: call(run, arguments))


def add_options(parser: argparse.ArgumentParser, run: Runner) -> None:
    """Add one option per keyword parameter of ``run`` to ``parser``.

    ``parser`` may be a subparser. Options are stored under the parameter's own
    name and carry no default, so ``call`` passes only what the user gave. A
    parameter bound by ``functools.partial`` gets no option.
    Help text comes from the ``Args:`` section of the runner's docstring.

    Raises:
        NameError: A parameter annotation names something the runner's module
            does not define at run time, such as an import under ``TYPE_CHECKING``.
    """
    function = _function(run)
    namespace: dict[str, object] = getattr(function, "__globals__", {})
    descriptions = argument_descriptions(inspect.getdoc(function) or "")
    bound = _bound(run)
    for parameter in inspect.signature(run).parameters.values():
        if parameter.kind in _KEYWORDS and parameter.name not in bound:
            _add_option(parser, parameter, namespace, descriptions.get(parameter.name, ""))


def call(run: Runner, arguments: argparse.Namespace) -> object:
    """Call ``run`` with the options ``add_options`` stored in ``arguments``."""
    values = vars(arguments)
    return run(**{
        name: values[name]
        for name, parameter in inspect.signature(run).parameters.items()
        if parameter.kind in _KEYWORDS and name in values
    })


def execute(prog: str, action: Callable[[], object]) -> int:
    """Run ``action``, print its result as JSON and return the exit status.

    Failures are reported exactly as ``main`` reports them, so hand-written
    subcommands behave like the generated one.
    """
    try:
        result = action()
    except _FAILURES as error:
        print(f"{prog}: {type(error).__name__}: {error}", file=sys.stderr)
        hint: object = getattr(error, "hint", None)
        if isinstance(hint, str) and hint:
            print(f"{prog}: hint: {hint}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def argument_descriptions(docstring: str) -> dict[str, str]:
    """Extract literal argument references and Args prose without inferring constraints.

    Introductory paragraphs are copied with whitespace normalized, stopping at the
    first section heading. Args entries replace this fallback, including grouped
    names wrapped across lines. Undocumented arguments retain no description.
    """
    descriptions: dict[str, str] = {}
    introduction: list[str] = []
    for line in docstring.splitlines():
        if re.fullmatch(r"[A-Za-z][\w ]*:", line):
            break
        introduction.append(line)
    for paragraph in re.split(r"\n\s*\n", "\n".join(introduction)):
        prose = " ".join(paragraph.split())
        referenced_names = {
            match.group("name") for match in re.finditer(
                r"(?<!`)(?P<ticks>`{1,2})(?P<name>[A-Za-z_]\w*)(?P=ticks)(?!`)", prose
            )
        }
        for name in referenced_names:
            previous = descriptions.get(name)
            descriptions[name] = f"{previous}\n\n{prose}" if previous else prose

    names: list[str] = []
    pending = ""
    in_args = False
    for line in docstring.splitlines():
        if line.strip() in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            names = []
            continue
        if not in_args:
            continue
        if line and not line[0].isspace():
            break
        if pending or re.match(r"^    \S", line):
            text = pending + line.strip()
            match = re.fullmatch(r"([\w, ]+)(?: \([^)]*\))?:\s*(.*)", text)
            if match:
                names = [name.strip() for name in match.group(1).split(",")]
                for name in names:
                    descriptions[name] = match.group(2).strip()
                pending = ""
                continue
            if re.fullmatch(r"(?:[A-Za-z_]\w*\s*,\s*)+", text):
                pending = text + " "
                names = []
                continue
            pending = ""
        if line.strip():
            for name in names:
                descriptions[name] += " " + line.strip()
    return descriptions


def _function(run: Runner) -> Runner:
    """Return the function under any ``functools.partial`` layers."""
    while isinstance(run, functools.partial):
        run = cast(Runner, run.func)
    return run


def _bound(run: Runner) -> frozenset[str]:
    """Return the keyword names fixed by ``functools.partial`` layers."""
    names: set[str] = set()
    while isinstance(run, functools.partial):
        names.update(run.keywords)
        run = cast(Runner, run.func)
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class _Shape:
    """How one annotation is parsed: ``kind`` plus the converter or choices it needs."""

    kind: Literal["flag", "choice", "scalar", "list", "map", "json"]
    convert: Converter = str
    choices: tuple[object, ...] = ()


def _add_option(
    parser: argparse.ArgumentParser,
    parameter: inspect.Parameter,
    namespace: dict[str, object],
    description: str,
) -> None:
    name = parameter.name
    flag = "--" + name.replace("_", "-")
    required = parameter.default is inspect.Parameter.empty
    shape = _shape(_annotation(parameter, namespace))
    default = ""
    if not required and parameter.default is not None and "default" not in description.lower():
        default = f" (default: {_shown(parameter.default)})"
    text = (description + default).strip().replace("%", "%%")
    unset = argparse.SUPPRESS
    match shape.kind:
        case "flag":
            parser.add_argument(
                flag, dest=name, action=argparse.BooleanOptionalAction, default=unset,
                required=required, help=text or None,
            )
        case "choice":
            parser.add_argument(
                flag, dest=name, type=shape.convert, choices=shape.choices, default=unset,
                required=required, help=text or None,
            )
        case "scalar" | "json":
            parser.add_argument(
                flag, dest=name, type=shape.convert, default=unset, required=required,
                metavar="JSON" if shape.kind == "json" else None, help=text or None,
            )
        case "list":
            parser.add_argument(
                flag, dest=name, type=shape.convert, nargs="+", default=unset,
                required=required, help=text or None,
            )
        case "map":
            parser.add_argument(
                flag, dest=name, type=shape.convert, action=_Entry, default=unset,
                required=required, metavar="KEY=VALUE", help=text or None,
            )


def _shown(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _annotation(parameter: inspect.Parameter, namespace: dict[str, object]) -> object:
    """Resolve one parameter's annotation in its runner's module, ignoring the others.

    Resolving only parameters keeps a return type imported under ``TYPE_CHECKING``
    from blocking the command.
    """
    if parameter.annotation is inspect.Parameter.empty:
        return object
    holder = SimpleNamespace(__annotations__={parameter.name: parameter.annotation})
    return get_type_hints(holder, globalns=namespace)[parameter.name]


def _members(annotation: object) -> tuple[list[object], bool]:
    """Flatten aliases and unions; return the non-``None`` members and whether ``None`` was one."""
    found: list[object] = []
    optional = False
    pending = [annotation]
    while pending:
        item = pending.pop(0)
        if isinstance(item, TypeAliasType):
            pending.insert(0, item.__value__)
        elif item is NoneType or item is None:
            optional = True
        elif get_origin(item) in (Union, UnionType):
            pending[0:0] = get_args(item)
        else:
            found.append(item)
    return found, optional


def _text_like(annotation: object) -> bool:
    return annotation is str or annotation is Path or annotation is os.PathLike \
        or get_origin(annotation) is os.PathLike


def _convert(members: list[object]) -> Converter | None:
    """Return the converter for text, path and number members, else ``None``."""
    if len(members) == 1 and members[0] is Path:
        return Path
    if members and all(_text_like(member) for member in members):
        return str
    if len(members) == 1 and (members[0] is int or members[0] is float):
        return cast(Converter, members[0])
    return None


def _scalar(annotation: object) -> Converter | None:
    """Return the converter for a required text or number element annotation."""
    members, optional = _members(annotation)
    return None if optional else _convert(members)


def _item(annotation: object) -> object | None:
    """Return the element type of ``Sequence[X]``, ``list[X]`` or ``tuple[X, ...]``."""
    origin = get_origin(annotation)
    arguments: tuple[object, ...] = get_args(annotation)
    if origin in (Sequence, list) and len(arguments) == 1:
        return arguments[0]
    if origin is tuple and len(arguments) == 2 and arguments[1] is Ellipsis:
        return arguments[0]
    return None


def _shape(annotation: object) -> _Shape:
    members, _ = _members(annotation)
    if members == [bool]:
        return _Shape("flag")
    if len(members) == 1 and get_origin(members[0]) is Literal:
        options: tuple[object, ...] = get_args(members[0])
        by_text = {str(option): option for option in options}
        return _Shape("choice", lambda text: by_text.get(text, text), options)
    scalar = _convert(members)
    if scalar is not None:
        return _Shape("scalar", scalar)
    if len(members) == 1:
        element = _item(members[0])
        converter = None if element is None else _scalar(element)
        if converter is not None:
            return _Shape("list", converter)
        origin = get_origin(members[0])
        arguments: tuple[object, ...] = get_args(members[0])
        if origin in (Mapping, dict) and len(arguments) == 2 and arguments[0] is str:
            value = _scalar(arguments[1])
            if value is not None:
                return _Shape("map", _entry(value))
    if len(members) == 2:
        for single, many in (members, members[::-1]):
            element = _item(many)
            converter = _convert([single])
            if element is not None and converter is not None and single == element:
                return _Shape("list", converter)
    return _Shape("json", _json)


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"{value} is not allowed; use finite numbers")


def _json(text: str) -> object:
    """Parse finite JSON; a leading ``@`` reads the JSON from that file."""
    try:
        if text.startswith("@"):
            text = Path(text[1:]).expanduser().read_text(encoding="utf-8")
        value: object = json.loads(text, parse_constant=_reject_constant)
        json.dumps(value, allow_nan=False)
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"not finite JSON: {error}") from None
    return value


def _entry(convert: Converter) -> Callable[[str], tuple[str, object]]:
    def parse(text: str) -> tuple[str, object]:
        key, separator, value = text.partition("=")
        if not separator or not key:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
        return key, convert(value)

    return parse


class _Entry(argparse.Action):
    """Collect repeated ``KEY=VALUE`` options into one dict and refuse a repeated key."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        key, value = cast(tuple[str, object], values)
        entries: dict[str, object] = dict(getattr(namespace, self.dest, None) or {})
        if key in entries:
            parser.error(f"argument {option_string}: {key!r} is given more than once")
        entries[key] = value
        setattr(namespace, self.dest, entries)
