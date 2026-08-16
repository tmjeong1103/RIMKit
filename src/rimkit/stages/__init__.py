"""Typed DMR and CoRe pipeline stages."""

from rimkit.stages.ara import (
    AraDiagnostics,
    AraFloorStats,
    AraResult,
    AraSlipStats,
    run_ara,
)
from rimkit.stages.diagnostics import (
    DiagnosticTrajectoriesResult,
    run_diagnostic_trajectories,
)
from rimkit.stages.dmr import (
    DmrProgress,
    DmrResult,
    build_base_orientation_targets,
    run_dmr,
)
from rimkit.stages.final_collision import (
    FinalCollisionDiagnostics,
    FinalCollisionProgress,
    FinalCollisionResult,
    run_final_collision,
)
from rimkit.stages.fpa import (
    FPA_IK_SOLVE_LABELS,
    FPA_TARGET_SOLVE_LABELS,
    FpaIkResult,
    FpaResult,
    FpaSolveRecord,
    FpaTargetsResult,
    build_fpa_targets,
    run_fpa,
    solve_fpa,
)
from rimkit.stages.initial_collision import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
    InitialCollisionProgress,
    InitialCollisionResult,
    run_initial_collision,
)
from rimkit.stages.target_trajectories import (
    TargetTrajectoriesResult,
    run_target_trajectories,
)

__all__ = [
    "AraDiagnostics",
    "AraFloorStats",
    "AraResult",
    "AraSlipStats",
    "CollisionPassDiagnostics",
    "DiagnosticTrajectoriesResult",
    "DmrProgress",
    "DmrResult",
    "InitialCollisionDiagnostics",
    "InitialCollisionProgress",
    "InitialCollisionResult",
    "FinalCollisionDiagnostics",
    "FinalCollisionProgress",
    "FinalCollisionResult",
    "FPA_IK_SOLVE_LABELS",
    "FPA_TARGET_SOLVE_LABELS",
    "FpaIkResult",
    "FpaResult",
    "FpaSolveRecord",
    "FpaTargetsResult",
    "TargetTrajectoriesResult",
    "build_base_orientation_targets",
    "build_fpa_targets",
    "run_ara",
    "run_dmr",
    "run_diagnostic_trajectories",
    "run_final_collision",
    "run_fpa",
    "run_initial_collision",
    "run_target_trajectories",
    "solve_fpa",
]
