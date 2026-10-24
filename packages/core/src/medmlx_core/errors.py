"""Shared MedMLX exceptions with builtin base classes."""

from __future__ import annotations


class MedmlxError(Exception):
    """Base class for the shared MedMLX runtime exception family."""

    def __str__(self) -> str:
        if not self.args:
            return ""
        if len(self.args) == 1:
            return str(self.args[0])
        return str(self.args)


class MissingDependencyError(MedmlxError, ImportError):
    """Raised when an optional model dependency/extra is not installed.

    Args:
        message: Human-readable description of the missing dependency.
        extra: The pip extra that provides the dependency (e.g. ``"conversion"``),
            or ``None`` when no single extra covers it.
        hint: An actionable install/setup hint, e.g.
            ``"Install medmlx-core[conversion]"``.

    Attributes:
        extra (str | None): The pip extra carrying the dependency.
        hint (str | None): Suggested install/setup command.
    """

    def __init__(
        self,
        message: str,
        *,
        extra: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.extra = extra
        self.hint = hint


class ModelExecutionError(MedmlxError, RuntimeError):
    """Raised when a model runner fails during execution.

    This signals a model/runtime failure, not a caller input error.
    """


class InvalidInputError(MedmlxError, ValueError):
    """Raised when caller-supplied inputs are missing, malformed, or an unsupported combination."""


class AssetNotReadyError(MedmlxError, FileNotFoundError):
    """Raised when a model's local assets or checkpoints are missing or mismatched.

    Args:
        message: Human-readable description of the missing or incompatible asset.
        reason: Short machine-friendly reason, e.g.
            ``"missing weights/model.safetensors"``.
        hint: An actionable setup hint, e.g. a model checkpoint conversion command.

    Attributes:
        reason (str | None): Why the asset is not ready.
        hint (str | None): Suggested conversion or setup command.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.hint = hint


__all__ = [
    "AssetNotReadyError",
    "InvalidInputError",
    "MedmlxError",
    "MissingDependencyError",
    "ModelExecutionError",
]
