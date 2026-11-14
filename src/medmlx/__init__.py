"""MedMLX model discovery and execution without importing model frameworks."""

from importlib.metadata import version

from medmlx.api import (
    DownloadProgress,
    ModelRun,
    ModelRunError,
    describe_model,
    list_checkpoints,
    list_models,
    list_packages,
    list_setup_plans,
    memory_status,
    model_readiness,
    nifti_shape,
    preview_output,
    run,
    setup_all,
    setup_model,
    setup_plan,
    start,
    start_cohort,
    summarize_output,
)
from medmlx.outputs import convert_dicom, list_dicom_series
from medmlx.probe import runtime_info

__version__ = version("medmlx")
__all__ = [
    "DownloadProgress",
    "ModelRun",
    "ModelRunError",
    "convert_dicom",
    "describe_model",
    "list_checkpoints",
    "list_dicom_series",
    "list_models",
    "list_packages",
    "list_setup_plans",
    "memory_status",
    "model_readiness",
    "nifti_shape",
    "preview_output",
    "run",
    "runtime_info",
    "setup_all",
    "setup_model",
    "setup_plan",
    "start",
    "start_cohort",
    "summarize_output",
]
