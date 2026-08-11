"""Trajectory helpers used by the research foot-placement adjustment path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.optimization.trajectory import same_length_jerk_matrix

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_SOLVERS = ("CLARABEL", "OSQP", "ECOS", "SCS")


@dataclass(frozen=True, slots=True)
class FpaSolveRecord:
    """Immutable provenance for one named FPA trajectory solve."""

    label: str
    solver: str
    status: str
    objective: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("FPA solve labels must be non-empty strings")
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError("FPA solver names must be non-empty strings")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("FPA solver statuses must be non-empty strings")
        label = self.label.strip()
        solver = self.solver.strip().upper()
        status = self.status.strip().lower()
        if status not in {"optimal", "optimal_inaccurate"}:
            raise ValueError(f"Unsupported successful FPA solver status: {status}")
        objective = float(self.objective)
        if not np.isfinite(objective):
            raise ValueError("FPA solver objectives must be finite")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "objective", objective)


@dataclass(frozen=True, slots=True)
class FpaTrajectoryResult:
    values: FloatArray
    status: str
    objective: float
    solver: str

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True).reshape(-1)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def _solve(problem: Any, preferred: str | None = "CLARABEL") -> str:
    candidates = (() if preferred is None else (preferred.upper(),)) + tuple(
        name for name in _SOLVERS if preferred is None or name != preferred.upper()
    )
    for name in candidates:
        solver = getattr(cp, name, None)
        if solver is None:
            continue
        try:
            problem.solve(solver=solver)
        except Exception:
            continue
        if str(problem.status) in {"optimal", "optimal_inaccurate"}:
            return name
    raise RuntimeError(f"FPA trajectory optimization failed (status={problem.status}).")


def _samples(seconds: ArrayLike, reference: ArrayLike) -> tuple[FloatArray, FloatArray, float]:
    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    if time_values.shape != reference_values.shape:
        raise ValueError("seconds and reference must have the same shape")
    if len(time_values) < 2:
        raise ValueError("at least two trajectory samples are required")
    if not np.isfinite(time_values).all() or not np.isfinite(reference_values).all():
        raise ValueError("trajectory samples must be finite")
    steps = np.diff(time_values)
    if np.any(steps <= 0.0):
        raise ValueError("seconds must be strictly increasing")
    minimum, maximum = float(np.min(steps)), float(np.max(steps))
    if minimum < 1e-9 or maximum / minimum > 1.01:
        raise ValueError("seconds must be uniformly sampled")
    return time_values, reference_values, float(np.mean(steps))


def shape_remain_trajectory(
    seconds: ArrayLike,
    reference: ArrayLike,
    remain_segments: Sequence[ArrayLike],
    *,
    jerk_weight: float,
    remain_weight: float,
) -> FpaTrajectoryResult:
    """Port ``traj_1d_remainer`` for its FPA piecewise-constant mode."""

    time_values, reference_values, dt = _samples(seconds, reference)
    count = len(time_values)
    jerk = same_length_jerk_matrix(count, dt)
    first = np.zeros((count - 1, count), dtype=np.float64)
    for index in range(count - 1):
        first[index, index] = -1.0
        first[index, index + 1] = 1.0

    variable = cp.Variable(count)
    objective: list[Any] = [
        cp.sum_squares(variable - reference_values),  # type: ignore[attr-defined]
        float(jerk_weight) * cp.sum_squares(jerk @ variable),  # type: ignore[attr-defined]
    ]
    for segment_value in remain_segments:
        segment = np.asarray(segment_value, dtype=np.int64).reshape(-1)
        if len(segment) == 0:
            continue
        start, end = int(segment[0]), int(segment[-1])
        if start < 0 or end >= count or start > end:
            raise IndexError(f"remain segment is outside the trajectory: {(start, end)}")
        if end > start:
            objective.append(
                float(remain_weight)
                * cp.sum_squares(  # type: ignore[attr-defined]
                    first[np.arange(start, end)] @ variable
                )
            )
    problem = cp.Problem(cp.Minimize(sum(objective)), [])
    solver = _solve(problem)
    if variable.value is None or problem.value is None:
        raise RuntimeError("FPA remain solver returned no solution")
    return FpaTrajectoryResult(
        values=np.asarray(variable.value, dtype=np.float64),
        status=str(problem.status),
        objective=float(problem.value),
        solver=solver,
    )


def shape_pinned_trajectory(
    seconds: ArrayLike,
    reference: ArrayLike,
    *,
    jerk_weight: float,
) -> FpaTrajectoryResult:
    """Squared tracking plus jerk regularization with fixed endpoints."""

    time_values, reference_values, dt = _samples(seconds, reference)
    jerk = same_length_jerk_matrix(len(time_values), dt)
    variable = cp.Variable(len(time_values))
    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(  # type: ignore[attr-defined]
                variable - reference_values
            )
            + float(jerk_weight) * cp.sum_squares(jerk @ variable)  # type: ignore[attr-defined]
        ),
        [
            variable[0] == float(reference_values[0]),
            variable[-1] == float(reference_values[-1]),
        ],
    )
    solver = _solve(problem)
    if variable.value is None or problem.value is None:
        raise RuntimeError("FPA pinned solver returned no solution")
    return FpaTrajectoryResult(
        values=np.asarray(variable.value, dtype=np.float64),
        status=str(problem.status),
        objective=float(problem.value),
        solver=solver,
    )


def _hermite_position(
    p0: FloatArray,
    p1: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    duration: float,
    phase: float,
) -> FloatArray:
    u = float(np.clip(phase, 0.0, 1.0))
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return np.asarray(
        h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1,
        dtype=np.float64,
    )


def _bounded_target(candidate: FloatArray, reference: FloatArray, maximum: float) -> FloatArray:
    delta = np.asarray(candidate) - np.asarray(reference)
    norm = float(np.linalg.norm(delta))
    if maximum > 0.0 and norm > maximum:
        delta *= float(maximum) / max(norm, 1e-12)
    return np.asarray(reference) + delta


def splice_contact_target_velocity(
    target_positions: ArrayLike,
    contact_segments: Sequence[ArrayLike],
    *,
    dt: float,
    touchdown_blend_time: float,
    touchdown_max_target_delta: float,
    release_blend_time: float,
    release_max_target_delta: float,
    max_transition_step: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Apply bounded Hermite touchdown and release splices."""

    source = np.asarray(target_positions, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("target_positions must have shape (frames, 3)")
    output = source.copy()
    gain = np.zeros(len(source), dtype=np.float64)
    if len(source) < 2:
        return output, gain
    segments = [
        np.asarray(value, dtype=np.int64).reshape(-1)
        for value in contact_segments
        if len(np.asarray(value).reshape(-1))
    ]
    segments.sort(key=lambda value: int(value[0]))
    if not segments:
        return output, gain

    velocity = np.gradient(source, float(dt), axis=0, edge_order=1)
    touchdown_frames = max(0, int(round(float(touchdown_blend_time) / float(dt))))
    release_frames = max(0, int(round(float(release_blend_time) / float(dt))))
    touchdown_starts: list[int] = []
    applied: list[tuple[str, int, int]] = []
    for index, segment in enumerate(segments):
        contact_start = int(segment[0])
        previous_end = int(segments[index - 1][-1]) if index else -1
        blend_start = max(previous_end + 1, contact_start - touchdown_frames)
        touchdown_starts.append(blend_start)
        if blend_start >= contact_start:
            gain[contact_start] = 1.0
            continue
        duration = max((contact_start - blend_start) * float(dt), 1e-12)
        p0, p1 = source[blend_start].copy(), source[contact_start].copy()
        v0, v1 = velocity[blend_start].copy(), np.zeros(3)
        tangent_limit = 3.0 * float(np.linalg.norm(p1 - p0)) / duration
        v0_norm = float(np.linalg.norm(v0))
        if v0_norm > tangent_limit:
            v0 *= tangent_limit / max(v0_norm, 1e-12)
        for tick in range(blend_start, contact_start + 1):
            phase = float(tick - blend_start) / float(contact_start - blend_start)
            output[tick] = _bounded_target(
                _hermite_position(p0, p1, v0, v1, duration, phase),
                source[tick],
                touchdown_max_target_delta,
            )
            gain[tick] = max(gain[tick], 3.0 * phase**2 - 2.0 * phase**3)
        applied.append(("touchdown", blend_start, contact_start))

    for index, segment in enumerate(segments):
        contact_end = int(segment[-1])
        next_start = touchdown_starts[index + 1] if index + 1 < len(segments) else len(source)
        blend_end = min(len(source) - 1, contact_end + release_frames, next_start - 1)
        if blend_end <= contact_end:
            gain[contact_end] = 1.0
            continue
        duration = max((blend_end - contact_end) * float(dt), 1e-12)
        p0, p1 = source[contact_end].copy(), source[blend_end].copy()
        v0, v1 = np.zeros(3), velocity[blend_end].copy()
        tangent_limit = 3.0 * float(np.linalg.norm(p1 - p0)) / duration
        v1_norm = float(np.linalg.norm(v1))
        if v1_norm > tangent_limit:
            v1 *= tangent_limit / max(v1_norm, 1e-12)
        for tick in range(contact_end, blend_end + 1):
            phase = float(tick - contact_end) / float(blend_end - contact_end)
            output[tick] = _bounded_target(
                _hermite_position(p0, p1, v0, v1, duration, phase),
                source[tick],
                release_max_target_delta,
            )
            gain[tick] = max(gain[tick], 1.0 - (3.0 * phase**2 - 2.0 * phase**3))
        applied.append(("release", contact_end, blend_end))
    for segment in segments:
        output[segment] = source[segment]
        gain[segment] = 1.0
    if max_transition_step is not None:
        maximum = float(max_transition_step)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_transition_step must be positive or None")
        for transition_type, start, end in applied:
            if transition_type == "touchdown":
                for tick in range(end - 1, start - 1, -1):
                    output[tick] = _bounded_target(output[tick], output[tick + 1], maximum)
            else:
                for tick in range(start + 1, end + 1):
                    output[tick] = _bounded_target(output[tick], output[tick - 1], maximum)
    return output, np.clip(gain, 0.0, 1.0)


def slew_limited_lower_envelope(
    values: ArrayLike,
    *,
    max_step: float,
    max_value: float,
) -> FloatArray:
    """Smooth a positive correction without exceeding its per-frame bound."""

    output = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_value))
    for tick in range(1, len(output)):
        output[tick] = min(output[tick], output[tick - 1] + float(max_step))
    for tick in range(len(output) - 2, -1, -1):
        output[tick] = min(output[tick], output[tick + 1] + float(max_step))
    return output


def slew_limited_upper_envelope(
    values: ArrayLike,
    *,
    max_step: float,
    max_value: float,
) -> FloatArray:
    """Spread a positive requirement without ever reducing it."""

    output = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_value))
    for tick in range(1, len(output)):
        output[tick] = max(output[tick], output[tick - 1] - float(max_step))
    for tick in range(len(output) - 2, -1, -1):
        output[tick] = max(output[tick], output[tick + 1] - float(max_step))
    return output


__all__ = [
    "FpaSolveRecord",
    "FpaTrajectoryResult",
    "shape_pinned_trajectory",
    "shape_remain_trajectory",
    "slew_limited_lower_envelope",
    "slew_limited_upper_envelope",
    "splice_contact_target_velocity",
]
