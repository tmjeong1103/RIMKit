"""Public entry points for RIMKit's CoRe retargeting method."""

from rimkit.api import Retargeter
from rimkit.config.schema import RunConfig
from rimkit.pipeline.runner import RetargetRunResult, run_retarget_pipeline

METHOD_ID = "core"

__all__ = [
    "METHOD_ID",
    "RetargetRunResult",
    "Retargeter",
    "RunConfig",
    "run_retarget_pipeline",
]
