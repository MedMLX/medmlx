"""Runtimes catch the shared errors as the built-in types they already handle."""

import pytest

import medmlx_core
from medmlx_core import errors


@pytest.mark.parametrize(
    ("name", "builtin"),
    [
        ("InvalidInputError", ValueError),
        ("MissingDependencyError", ImportError),
        ("ModelExecutionError", RuntimeError),
        ("AssetNotReadyError", FileNotFoundError),
    ],
)
def test_shared_errors_are_public_and_catchable_as_builtins(
    name: str, builtin: type[Exception],
) -> None:
    cls = getattr(medmlx_core, name)
    assert cls is getattr(errors, name)
    assert issubclass(cls, errors.MedmlxError) and issubclass(cls, builtin)
