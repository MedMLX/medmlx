"""Public exception identity and built-in catch contracts."""

import pytest

import medmlx_core
from medmlx_core import errors


@pytest.mark.parametrize(
    "name,builtin",
    [
        ("InvalidInputError", ValueError),
        ("MissingDependencyError", ImportError),
        ("ModelExecutionError", RuntimeError),
        ("AssetNotReadyError", FileNotFoundError),
    ],
)
def test_shared_error_base(name: str, builtin: type[Exception]) -> None:
    cls = getattr(errors, name)
    exc = cls("message")
    assert isinstance(exc, errors.MedmlxError)
    assert isinstance(exc, builtin)
    assert getattr(medmlx_core, name) is cls
    assert exc.args == ("message",)
    assert str(exc) == "message"


def test_medmlx_error_is_the_real_public_base() -> None:
    assert errors.MedmlxError.__name__ == "MedmlxError"
    assert errors.MedmlxError.__bases__ == (Exception,)
    assert medmlx_core.MedmlxError is errors.MedmlxError
    assert str(errors.MedmlxError()) == ""
    assert str(errors.MedmlxError("message", "detail")) == "('message', 'detail')"


def test_error_metadata() -> None:
    missing = errors.MissingDependencyError(
        "missing", extra="conversion", hint="install conversion"
    )
    assert vars(missing) == {"extra": "conversion", "hint": "install conversion"}
    asset = errors.AssetNotReadyError("absent", reason="missing weights", hint="convert weights")
    assert vars(asset) == {"reason": "missing weights", "hint": "convert weights"}
    assert vars(errors.MissingDependencyError("missing")) == {"extra": None, "hint": None}
    assert vars(errors.AssetNotReadyError("absent")) == {"reason": None, "hint": None}
