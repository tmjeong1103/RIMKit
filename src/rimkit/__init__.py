"""RIMKit public package interface."""

from rimkit._version import __version__
from rimkit.api import PreflightResult, Retargeter
from rimkit.config.schema import RunConfig
from rimkit.exceptions import RIMKitError
from rimkit.methods import MethodSpec, get_method, list_methods
from rimkit.pipeline.runner import RetargetRunResult
from rimkit.review import ReviewRunResult, run_review
from rimkit.robots.registry import get_robot, list_robots
from rimkit.stages.diagnostics import DiagnosticTrajectoriesResult
from rimkit.stages.dmr import DmrProgress, DmrResult
from rimkit.stages.final_collision import (
    FinalCollisionDiagnostics,
    FinalCollisionProgress,
    FinalCollisionResult,
)
from rimkit.stages.fpa import (
    FpaIkResult,
    FpaResult,
    FpaSolveRecord,
    FpaTargetsResult,
)
from rimkit.stages.initial_collision import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
    InitialCollisionProgress,
    InitialCollisionResult,
)
from rimkit.stages.target_trajectories import TargetTrajectoriesResult

__all__ = [
    "DmrProgress",
    "DmrResult",
    "DiagnosticTrajectoriesResult",
    "CollisionPassDiagnostics",
    "InitialCollisionDiagnostics",
    "InitialCollisionProgress",
    "InitialCollisionResult",
    "MethodSpec",
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
    "RIMKitError",
    "RunConfig",
    "__version__",
    "get_robot",
    "get_method",
    "list_methods",
    "list_robots",
    "run_review",
]
