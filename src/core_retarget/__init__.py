"""CoRe public package interface."""

from core_retarget._version import __version__
from core_retarget.api import PreflightResult, Retargeter
from core_retarget.config.schema import RunConfig
from core_retarget.pipeline.runner import RetargetRunResult
from core_retarget.review import ReviewRunResult, run_review
from core_retarget.robots.registry import get_robot, list_robots
from core_retarget.stages.diagnostics import DiagnosticTrajectoriesResult
from core_retarget.stages.dmr import DmrProgress, DmrResult
from core_retarget.stages.final_collision import (
    FinalCollisionDiagnostics,
    FinalCollisionProgress,
    FinalCollisionResult,
)
from core_retarget.stages.fpa import (
    FpaIkResult,
    FpaResult,
    FpaSolveRecord,
    FpaTargetsResult,
)
from core_retarget.stages.initial_collision import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
    InitialCollisionProgress,
    InitialCollisionResult,
)
from core_retarget.stages.target_trajectories import TargetTrajectoriesResult

__all__ = [
    "DmrProgress",
    "DmrResult",
    "DiagnosticTrajectoriesResult",
    "CollisionPassDiagnostics",
    "InitialCollisionDiagnostics",
    "InitialCollisionProgress",
    "InitialCollisionResult",
    "FinalCollisionDiagnostics",
    "FinalCollisionProgress",
    "FinalCollisionResult",
    "FpaIkResult",
    "FpaResult",
    "FpaSolveRecord",
    "FpaTargetsResult",
    "TargetTrajectoriesResult",
    "PreflightResult",
    "Retargeter",
    "RetargetRunResult",
    "ReviewRunResult",
    "RunConfig",
    "__version__",
    "get_robot",
    "list_robots",
    "run_review",
]
