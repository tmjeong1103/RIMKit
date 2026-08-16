"""Joint-trajectory shaping used by initial collision refinement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_SUCCESS_STATUSES = frozenset(("optimal", "optimal_inaccurate"))
_FALLBACK_SOLVERS = ("CLARABEL", "OSQP", "ECOS", "SCS")


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _first_difference_matrix(sample_count: int, sample_period: float) -> FloatArray:
    """Return the private first-derivative stencil used for short trajectories."""

    matrix = np.zeros((sample_count, sample_count), dtype=np.float64)
    if sample_count == 1:
        return matrix

    inverse_period = 1.0 / sample_period
    matrix[0, 0] = -inverse_period
    matrix[0, 1] = inverse_period
    if sample_count == 2:
        matrix[1, 0] = -inverse_period
        matrix[1, 1] = inverse_period
        return matrix

    for row in range(1, sample_count - 1):
        matrix[row, row - 1] = -0.5 * inverse_period
        matrix[row, row + 1] = 0.5 * inverse_period
    matrix[-1, -2] = -inverse_period
    matrix[-1, -1] = inverse_period
    return matrix


def _set_forward_jerk_row(
    matrix: FloatArray,
    row: int,
    first_column: int,
    scale: float,
) -> None:
    matrix[row, first_column : first_column + 4] = (-scale, 3.0 * scale, -3.0 * scale, scale)


def _set_backward_jerk_row(
    matrix: FloatArray,
    row: int,
    first_column: int,
    scale: float,
) -> None:
    matrix[row, first_column : first_column + 4] = (scale, -3.0 * scale, 3.0 * scale, -scale)


def same_length_jerk_matrix(sample_count: int, sample_period: float) -> FloatArray:
    """Return the same-length third-difference matrix used by CoRe.

    The output always has shape ``(sample_count, sample_count)``.  For one to
    three samples, the research code falls back to applying its first-
    derivative matrix three times.  Four or more samples use its original
    one-sided and five-point stencils, including their established boundary
    signs.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be an integer.")
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive.")
    try:
        period = float(sample_period)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample_period must be a finite positive scalar.") from exc
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be a finite positive scalar.")

    matrix = np.zeros((count, count), dtype=np.float64)
    if count <= 3:
        first = _first_difference_matrix(count, period)
        return _readonly_float(first @ first @ first)

    forward_scale = 1.0 / (period * period * period)
    if count == 4:
        _set_forward_jerk_row(matrix, 0, 0, forward_scale)
        _set_forward_jerk_row(matrix, 1, 0, forward_scale)
        _set_backward_jerk_row(matrix, 2, 0, forward_scale)
        _set_backward_jerk_row(matrix, 3, 0, forward_scale)
        matrix.setflags(write=False)
        return matrix

    central_scale = 0.5 * forward_scale
    _set_forward_jerk_row(matrix, 0, 0, forward_scale)
    _set_forward_jerk_row(matrix, 1, 1, forward_scale)
    for row in range(2, count - 2):
        matrix[row, row - 2] = central_scale
        matrix[row, row - 1] = -2.0 * central_scale
        matrix[row, row + 1] = 2.0 * central_scale
        matrix[row, row + 2] = -central_scale
    _set_backward_jerk_row(matrix, count - 2, count - 5, forward_scale)
    _set_backward_jerk_row(matrix, count - 1, count - 4, forward_scale)
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class TrajectoryShapeResult:
    """Immutable solution of one unconstrained joint-trajectory shaping problem."""

    values: FloatArray
    status: str
    objective: float
    solver: str

    def __post_init__(self) -> None:
        values = _readonly_float(self.values).reshape(-1)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def z(self) -> FloatArray:
        """Compatibility alias for the optimized trajectory."""

        return self.values


def _try_cvxpy_solver(
    problem: Any,
    solver_name: str,
    solver_options: Mapping[str, Any],
) -> bool:
    name = solver_name.upper()
    solver = getattr(cp, name, None)
    if solver is None:
        return False
    try:
        problem.solve(solver=solver, **dict(solver_options))
    except Exception:
        return False
    return str(problem.status) in _SUCCESS_STATUSES


def shape_trajectory_1d(
    seconds: ArrayLike,
    reference: ArrayLike,
    *,
    tracking_norm: int = 1,
    jerk_weight: float = 1e-5,
    preferred_solver: str | None = "CLARABEL",
    solver_options: Mapping[str, Any] | None = None,
) -> TrajectoryShapeResult:
    """Shape one joint trajectory with an unconstrained jerk objective.

    The objective is ``||z-reference||_p + jerk_weight*||A_jerk z||_2^2``.
    ``p`` may be one or two; the two-norm case uses the squared two-norm.
    Solver selection uses the following deterministic order:
    the preferred solver is tried first, then CLARABEL, OSQP, ECOS, and SCS.
    """

    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    if time_values.shape != reference_values.shape:
        raise ValueError("seconds and reference must have the same shape.")
    if time_values.size < 2:
        raise ValueError("seconds and reference must contain at least two samples.")
    if not np.isfinite(time_values).all():
        raise ValueError("seconds must contain only finite values.")
    if not np.isfinite(reference_values).all():
        raise ValueError("reference must contain only finite values.")
    if tracking_norm not in (1, 2):
        raise ValueError("tracking_norm must be 1 or 2.")
    try:
        weight = float(jerk_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("jerk_weight must be a finite non-negative scalar.") from exc
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("jerk_weight must be a finite non-negative scalar.")
    if preferred_solver is not None and not isinstance(preferred_solver, str):
        raise TypeError("preferred_solver must be a string or None.")

    time_steps = np.diff(time_values)
    if np.any(time_steps <= 0.0):
        raise ValueError("seconds must be strictly increasing.")
    minimum_step = float(np.min(time_steps))
    maximum_step = float(np.max(time_steps))
    if minimum_step < 1e-9:
        raise ValueError(f"seconds step is too small ({minimum_step:g}).")
    if maximum_step / minimum_step > 1.01:
        raise ValueError(
            "seconds must be uniformly sampled within one percent "
            f"(min={minimum_step:g}, max={maximum_step:g})."
        )
    sample_period = float(np.mean(time_steps))
    jerk_matrix = same_length_jerk_matrix(len(time_values), sample_period)

    variable = cp.Variable(len(time_values))
    tracking_mask = np.ones(len(time_values), dtype=np.float64)
    tracking_error = cp.multiply(  # type: ignore[attr-defined]
        tracking_mask,
        variable - reference_values,
    )
    if tracking_norm == 1:
        tracking_objective = cp.norm1(tracking_error)  # type: ignore[attr-defined]
    else:
        tracking_objective = cp.sum_squares(tracking_error)  # type: ignore[attr-defined]
    objective_terms = [
        tracking_objective,
        weight * cp.sum_squares(jerk_matrix @ variable),  # type: ignore[attr-defined]
    ]
    problem = cp.Problem(cp.Minimize(sum(objective_terms)), [])
    options: Mapping[str, Any] = {} if solver_options is None else dict(solver_options)

    solver_used: str | None = None
    if preferred_solver is not None and _try_cvxpy_solver(problem, preferred_solver, options):
        solver_used = preferred_solver.upper()
    if solver_used is None:
        preferred_upper = preferred_solver.upper() if preferred_solver is not None else None
        for candidate in _FALLBACK_SOLVERS:
            if candidate == preferred_upper:
                continue
            if _try_cvxpy_solver(problem, candidate, options):
                solver_used = candidate
                break
    if solver_used is None:
        raise RuntimeError(f"Trajectory shaping failed (status={problem.status}).")

    if variable.value is None or problem.value is None:
        raise RuntimeError("Trajectory shaping solver returned no solution.")
    values = np.asarray(variable.value, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or not np.isfinite(float(problem.value)):
        raise RuntimeError("Trajectory shaping solver returned a non-finite solution.")
    return TrajectoryShapeResult(
        values=values,
        status=str(problem.status),
        objective=float(problem.value),
        solver=solver_used,
    )


__all__ = [
    "TrajectoryShapeResult",
    "same_length_jerk_matrix",
    "shape_trajectory_1d",
]
