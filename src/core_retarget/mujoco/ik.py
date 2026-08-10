"""MuJoCo body-point inverse kinematics with native acceleration.

Robot profiles turn semantic retargeting constraints into world-space point
targets; this module solves those targets against a MuJoCo model. The Python
and packaged C++ backends use the same numerical loop: weighted point
Jacobians, scale-normalized damped least squares, an optional free-base update
before the articulated-joint update, a
joint-limit probe followed by a reduced re-solve, and selection of the best of
``max_iterations + 1`` candidate poses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.native import (
    BackendPreference,
    BackendSelection,
    mujoco_addresses,
    resolve_backend,
)

FloatArray = NDArray[np.float64]


@runtime_checkable
class MuJoCoModelAdapter(Protocol):
    """Structural interface required by :class:`BodyPositionIKSolver`."""

    model: mujoco.MjModel
    data: mujoco.MjData

    @property
    def rev_joint_names(self) -> Sequence[str]:
        """Revolute joints in the adapter's canonical order."""

    @property
    def pri_joint_names(self) -> Sequence[str]:
        """Prismatic joints in the adapter's canonical order."""

    @property
    def rev_pri_joint_names(self) -> Sequence[str]:
        """Revolute joints followed by prismatic joints."""

    def copy_state_from(self, other: Any) -> None:
        """Copy a compatible model state and run forward kinematics."""


@dataclass(frozen=True)
class IkResult:
    """Best pose and convergence history from one IK solve.

    ``errors`` contains the initial candidate followed by one candidate for each
    update.  Consequently, its length is always ``iterations + 1``.  The solver
    deliberately leaves its internal MuJoCo state at the final candidate;
    callers should apply ``qpos`` when they
    need the selected best candidate.
    """

    qpos: FloatArray
    joint_qpos: FloatArray
    error: float
    iterations: int
    errors: FloatArray
    best_iteration: int
    joints: tuple[str, ...]


@dataclass(frozen=True)
class _PointTarget:
    body_id: int
    local_position: FloatArray
    target_position: FloatArray
    weight: float


def _trim_scale(values: FloatArray, threshold: float) -> FloatArray:
    """Scale a vector so its largest absolute component is at most a threshold."""

    result = values.copy()
    if result.size == 0:
        return result
    maximum = float(np.max(np.abs(result)))
    if maximum > threshold:
        result *= threshold / maximum
    return result


def _normalize_damping(damping: float | ArrayLike, size: int) -> FloatArray:
    if np.isscalar(damping):
        values = np.full(size, float(np.asarray(damping).item()), dtype=np.float64)
    else:
        values = np.asarray(damping, dtype=np.float64).reshape(-1)
        if len(values) != size:
            raise ValueError(f"damping has length {len(values)}; expected {size}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("damping must contain finite, non-negative values")
    return np.maximum(values, 1e-12)


def _solve_dls(jacobian: FloatArray, error: FloatArray, damping: FloatArray) -> FloatArray:
    """Solve the scale-normalized primal DLS system."""

    rows, columns = jacobian.shape
    if rows == 0 or columns == 0:
        return np.zeros(columns, dtype=np.float64)
    if not np.isfinite(jacobian).all() or not np.isfinite(error).all():
        return np.zeros(columns, dtype=np.float64)

    scale = float(np.max(np.abs(jacobian)))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros(columns, dtype=np.float64)

    scaled_jacobian = jacobian / scale
    scaled_error = error / scale
    scaled_damping = np.maximum(damping / (scale * scale), 1e-12)
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hessian = scaled_jacobian.T @ scaled_jacobian + np.diag(scaled_damping)
            rhs = scaled_jacobian.T @ scaled_error
            result = np.linalg.solve(hessian, rhs)
    except (FloatingPointError, np.linalg.LinAlgError):
        result = np.zeros(columns, dtype=np.float64)
    if not np.isfinite(result).all():
        return np.zeros(columns, dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _solve_linear_reference(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve with the scalar elimination order used by the C++ backend."""

    size = len(rhs)
    work = np.array(matrix, dtype=np.float64, copy=True, order="C")
    work_rhs = np.array(rhs, dtype=np.float64, copy=True).reshape(-1)
    result = np.zeros(size, dtype=np.float64)
    for column in range(size):
        pivot = column
        pivot_abs = abs(float(work[column, column]))
        for row in range(column + 1, size):
            value_abs = abs(float(work[row, column]))
            if value_abs > pivot_abs:
                pivot = row
                pivot_abs = value_abs
        if not math.isfinite(pivot_abs) or pivot_abs < 1e-18:
            return np.zeros(size, dtype=np.float64)
        if pivot != column:
            column_values = work[column, column:].copy()
            work[column, column:] = work[pivot, column:]
            work[pivot, column:] = column_values
            work_rhs[column], work_rhs[pivot] = work_rhs[pivot], work_rhs[column]

        diagonal = float(work[column, column])
        for row in range(column + 1, size):
            factor = float(work[row, column]) / diagonal
            work[row, column] = 0.0
            work[row, column + 1 :] -= factor * work[column, column + 1 :]
            work_rhs[row] -= factor * work_rhs[column]

    for row in range(size - 1, -1, -1):
        value = float(work_rhs[row])
        for column in range(row + 1, size):
            value -= float(work[row, column]) * float(result[column])
        diagonal = float(work[row, row])
        if not math.isfinite(diagonal) or abs(diagonal) < 1e-18:
            return np.zeros(size, dtype=np.float64)
        result[row] = value / diagonal
    if not np.isfinite(result).all():
        return np.zeros(size, dtype=np.float64)
    return result


def _solve_dls_reference(
    jacobian: FloatArray,
    error: FloatArray,
    damping: FloatArray,
) -> FloatArray:
    """Accumulate the DLS system in the native backend's row/a/b order."""

    rows, columns = jacobian.shape
    if rows == 0 or columns == 0:
        return np.zeros(columns, dtype=np.float64)
    if not (
        np.isfinite(jacobian).all() and np.isfinite(error).all() and np.isfinite(damping).all()
    ):
        return np.zeros(columns, dtype=np.float64)

    scale = 0.0
    for value in jacobian.reshape(-1):
        scale = max(scale, abs(float(value)))
    if not math.isfinite(scale) or scale < 1e-12:
        return np.zeros(columns, dtype=np.float64)

    inverse_scale = 1.0 / scale
    hessian = np.zeros((columns, columns), dtype=np.float64)
    rhs = np.zeros(columns, dtype=np.float64)
    for row in range(rows):
        error_scaled = float(error[row]) * inverse_scale
        scaled_row = np.asarray(jacobian[row], dtype=np.float64) * inverse_scale
        rhs += scaled_row * error_scaled
        hessian += np.multiply.outer(scaled_row, scaled_row)
    for column in range(columns):
        damping_value = float(damping[column]) / (scale * scale)
        if not math.isfinite(damping_value) or damping_value < 1e-12:
            damping_value = 1e-12
        hessian[column, column] += damping_value
    return _solve_linear_reference(hessian, rhs)


def _mask_task_irrelevant(jacobian: FloatArray, delta: FloatArray) -> FloatArray:
    result = delta.copy()
    if jacobian.shape[0] == 0 or jacobian.shape[1] == 0:
        return np.zeros_like(result)
    norms = np.linalg.norm(jacobian, axis=0)
    threshold = max(1e-12, 1e-6 * float(np.max(norms)))
    result[norms <= threshold] = 0.0
    return result


def _mask_task_irrelevant_reference(
    jacobian: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    """Apply the native task-column mask using its scalar accumulation order."""

    rows, columns = jacobian.shape
    result = delta.copy()
    column_norms = np.zeros(columns, dtype=np.float64)
    for column in range(columns):
        squared = 0.0
        for row in range(rows):
            value = float(jacobian[row, column])
            squared += value * value
        column_norms[column] = math.sqrt(squared)
    maximum = 0.0
    for value in column_norms:
        maximum = max(maximum, abs(float(value)))
    threshold = max(1e-12, 1e-6 * maximum)
    for column in range(columns):
        if float(column_norms[column]) <= threshold:
            result[column] = 0.0
    return result


def _project_nullspace(
    jacobian: FloatArray,
    delta: FloatArray,
    damping: FloatArray,
) -> FloatArray:
    if jacobian.shape[0] == 0 or jacobian.shape[1] == 0:
        return np.zeros_like(delta)
    scale = float(np.max(np.abs(jacobian)))
    if not np.isfinite(scale) or scale < 1e-12:
        return delta.copy()
    scaled_jacobian = jacobian / scale
    scaled_damping = np.maximum(damping / (scale * scale), 1e-12)
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hessian = scaled_jacobian.T @ scaled_jacobian + np.diag(scaled_damping)
            rhs = scaled_jacobian.T @ (scaled_jacobian @ delta)
            result = delta - np.linalg.solve(hessian, rhs)
    except (FloatingPointError, np.linalg.LinAlgError):
        result = np.zeros_like(delta)
    if not np.isfinite(result).all():
        return np.zeros_like(delta)
    return np.asarray(result, dtype=np.float64)


def _project_nullspace_reference(
    jacobian: FloatArray,
    delta: FloatArray,
    damping: FloatArray,
) -> FloatArray:
    """Project a home update with the native backend's scalar solve order."""

    rows, columns = jacobian.shape
    if rows == 0 or columns == 0:
        return np.zeros_like(delta)
    if not (
        np.isfinite(jacobian).all() and np.isfinite(delta).all() and np.isfinite(damping).all()
    ):
        return np.zeros_like(delta)

    scale = 0.0
    for value in jacobian.reshape(-1):
        scale = max(scale, abs(float(value)))
    if not math.isfinite(scale) or scale < 1e-12:
        return delta.copy()
    inverse_scale = 1.0 / scale
    hessian = np.zeros((columns, columns), dtype=np.float64)
    rhs = np.zeros(columns, dtype=np.float64)
    for row in range(rows):
        projected = 0.0
        for column in range(columns):
            projected += float(jacobian[row, column]) * inverse_scale * float(delta[column])
        scaled_row = np.asarray(jacobian[row], dtype=np.float64) * inverse_scale
        rhs += scaled_row * projected
        hessian += np.multiply.outer(scaled_row, scaled_row)
    for column in range(columns):
        damping_value = float(damping[column]) / (scale * scale)
        if not math.isfinite(damping_value) or damping_value < 1e-12:
            damping_value = 1e-12
        hessian[column, column] += damping_value
    correction = _solve_linear_reference(hessian, rhs)
    result = np.empty(columns, dtype=np.float64)
    for column in range(columns):
        result[column] = float(delta[column]) - float(correction[column])
    if not np.isfinite(result).all():
        return np.zeros_like(delta)
    return result


class BodyPositionIKSolver:
    """Damped-least-squares IK over points rigidly attached to MuJoCo bodies.

    Free roots, hinge joints, and slide joints are supported. MuJoCo polynomial
    joint equalities are not silently approximated: a model containing one is
    rejected until coupling support is added to this implementation.
    """

    def __init__(
        self,
        model_adapter: MuJoCoModelAdapter,
        *,
        max_iterations: int = 10,
        revolute_step: float = np.deg2rad(3.0),
        revolute_update_limit: float = np.deg2rad(5.0),
        damping: float | ArrayLike = 1e-6,
        joint_limit_probe: float = np.deg2rad(3.0),
        home: ArrayLike | None = None,
        prismatic_step: float = 0.05,
        prismatic_update_limit: float = 0.05,
        prismatic_limit_probe: float = 0.01,
        nullspace_gain: float = 0.5,
        reference_ordered: bool = False,
        backend: BackendPreference | BackendSelection = "python",
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        scalar_values = {
            "revolute_step": revolute_step,
            "revolute_update_limit": revolute_update_limit,
            "joint_limit_probe": joint_limit_probe,
            "prismatic_step": prismatic_step,
            "prismatic_update_limit": prismatic_update_limit,
            "prismatic_limit_probe": prismatic_limit_probe,
            "nullspace_gain": nullspace_gain,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.adapter = model_adapter
        self.model = model_adapter.model
        self.data = model_adapter.data
        self.max_iterations = int(max_iterations)
        # Mutable on purpose: the pelvis post-pass temporarily sets this
        # to zero while allowing only the free base to move.
        self.revolute_step = float(revolute_step)
        self.revolute_update_limit = float(revolute_update_limit)
        self.prismatic_step = float(prismatic_step)
        self.prismatic_update_limit = float(prismatic_update_limit)
        self.joint_limit_probe = float(joint_limit_probe)
        self.prismatic_limit_probe = float(prismatic_limit_probe)
        self.nullspace_gain = float(nullspace_gain)
        self.reference_ordered = bool(reference_ordered)
        self.backend = resolve_backend(backend)
        self._damping_input = damping

        self.joint_names = tuple(model_adapter.rev_pri_joint_names)
        if not self.joint_names:
            raise ValueError("model adapter has no revolute or prismatic joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("rev_pri_joint_names contains duplicates")

        rev_names = set(model_adapter.rev_joint_names)
        pri_names = set(model_adapter.pri_joint_names)
        unknown_types = set(self.joint_names) - rev_names - pri_names
        if unknown_types:
            raise ValueError(f"joint type is unavailable for: {sorted(unknown_types)}")
        self._is_revolute = np.asarray(
            [name in rev_names for name in self.joint_names], dtype=np.bool_
        )

        joint_ids: list[int] = []
        qpos_indices: list[int] = []
        dof_indices: list[int] = []
        for name in self.joint_names:
            joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ValueError(f"MuJoCo model has no joint named {name!r}")
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"joint {name!r} is not hinge or slide")
            joint_ids.append(joint_id)
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            dof_indices.append(int(self.model.jnt_dofadr[joint_id]))

        self._joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self._qpos_indices = np.asarray(qpos_indices, dtype=np.int32)
        self._dof_indices = np.asarray(dof_indices, dtype=np.int32)
        limited = np.asarray(self.model.jnt_limited[self._joint_ids], dtype=np.bool_)
        ranges = np.asarray(self.model.jnt_range[self._joint_ids], dtype=np.float64)
        self._minimum = ranges[:, 0].copy()
        self._maximum = ranges[:, 1].copy()
        self._minimum[~limited] = -np.inf
        self._maximum[~limited] = np.inf

        self._reject_joint_equalities()
        self._damping = _normalize_damping(damping, len(self.joint_names))
        self._home = None if home is None else np.asarray(home, dtype=np.float64).reshape(-1)
        if self._home is not None and len(self._home) != len(self.joint_names):
            raise ValueError(f"home has length {len(self._home)}; expected {len(self.joint_names)}")
        if self._home is not None and not np.isfinite(self._home).all():
            raise ValueError("home must contain only finite values")

        self._targets: list[_PointTarget] = []
        mujoco.mj_forward(self.model, self.data)

    @property
    def target_count(self) -> int:
        """Number of point constraints currently buffered."""

        return len(self._targets)

    def configure_nullspace(
        self,
        home: ArrayLike | None,
        *,
        gain: float,
    ) -> None:
        """Set the joint-space reference used by subsequent null-space solves.

        The G1/H2/R1 DMR paths change their reference from the neutral pose to
        the previously realized frame.  Keeping that operation public avoids
        reaching into solver internals while preserving the original solver's
        per-frame configuration semantics.
        """

        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        normalized_home = None if home is None else np.asarray(home, dtype=np.float64).reshape(-1)
        if normalized_home is not None:
            if len(normalized_home) != len(self.joint_names):
                raise ValueError(
                    f"home has length {len(normalized_home)}; expected {len(self.joint_names)}"
                )
            if not np.isfinite(normalized_home).all():
                raise ValueError("home must contain only finite values")
            normalized_home = normalized_home.copy()
        self.nullspace_gain = float(gain)
        self._home = normalized_home

    def _reject_joint_equalities(self) -> None:
        for equality_id in range(int(self.model.neq)):
            if self.model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_JOINT:
                raise NotImplementedError(
                    "BodyPositionIKSolver does not support MuJoCo joint equalities"
                )

    def _sync_from(self, source: MuJoCoModelAdapter) -> None:
        if source.model.nq != self.model.nq or source.model.nv != self.model.nv:
            raise ValueError("source model is not kinematically compatible with the IK model")
        self.adapter.copy_state_from(source)

    def reset_targets(self, sync_from: MuJoCoModelAdapter | None = None) -> None:
        """Clear point targets, optionally synchronizing model state first."""

        if sync_from is not None:
            self._sync_from(sync_from)
        self._targets.clear()

    def _body_id(self, body_name: str) -> int:
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id < 0:
            raise ValueError(f"MuJoCo model has no body named {body_name!r}")
        return body_id

    def _world_from_local(self, body_id: int, local_position: FloatArray) -> FloatArray:
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        origin = np.asarray(self.data.xpos[body_id], dtype=np.float64)
        if self.reference_ordered:
            result = np.empty(3, dtype=np.float64)
            for row in range(3):
                result[row] = float(origin[row])
                for column in range(3):
                    result[row] += float(rotation[row, column]) * float(local_position[column])
            return result
        return np.asarray(origin + rotation @ local_position, dtype=np.float64)

    def add_target(
        self,
        body_name: str,
        current_position: ArrayLike,
        target_position: ArrayLike,
        weight: float = 1.0,
    ) -> None:
        """Attach a current world point to a body and give it a world target."""

        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")
        current = np.asarray(current_position, dtype=np.float64).reshape(3)
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError("target positions must contain only finite values")
        body_id = self._body_id(body_name)
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        origin = np.asarray(self.data.xpos[body_id], dtype=np.float64)
        local = rotation.T @ (current - origin)
        self._targets.append(
            _PointTarget(
                body_id=body_id,
                local_position=np.asarray(local, dtype=np.float64),
                target_position=target.copy(),
                weight=float(weight),
            )
        )

    def add_transform_target(
        self,
        body_name: str,
        current_transform: ArrayLike,
        target_transform: ArrayLike,
        weight: float = 1.0,
        axis_length: float = 0.1,
    ) -> None:
        """Constrain a body transform using three axis endpoint point targets."""

        if not np.isfinite(axis_length) or axis_length < 0.0:
            raise ValueError("axis_length must be finite and non-negative")
        current = np.asarray(current_transform, dtype=np.float64)
        target = np.asarray(target_transform, dtype=np.float64)
        if current.shape != (4, 4) or target.shape != (4, 4):
            raise ValueError("transforms must have shape (4, 4)")
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError("transforms must contain only finite values")
        for axis in range(3):
            current_point = current[:3, 3] + axis_length * current[:3, axis]
            target_point = target[:3, 3] + axis_length * target[:3, axis]
            self.add_target(body_name, current_point, target_point, weight)

    def _build_weighted_system(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self.reference_ordered:
            row_count = 3 * len(self._targets)
            jacobian = np.zeros((row_count, self.model.nv), dtype=np.float64)
            weighted_error = np.zeros(row_count, dtype=np.float64)
            raw_error = np.zeros(row_count, dtype=np.float64)
            for target_index, target in enumerate(self._targets):
                current = self._world_from_local(target.body_id, target.local_position)
                error = target.target_position - current
                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    current,
                    target.body_id,
                )
                weight_scale = math.sqrt(target.weight)
                first_row = 3 * target_index
                last_row = first_row + 3
                raw_error[first_row:last_row] = error
                weighted_error[first_row:last_row] = weight_scale * error
                jacobian[first_row:last_row] = weight_scale * jacobian_position
            return jacobian, weighted_error, raw_error

        jacobian_blocks: list[FloatArray] = []
        weighted_errors: list[FloatArray] = []
        raw_errors: list[FloatArray] = []
        for target in self._targets:
            current = self._world_from_local(target.body_id, target.local_position)
            error = target.target_position - current
            jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                current,
                target.body_id,
            )
            weight_scale = np.sqrt(target.weight)
            jacobian_blocks.append(weight_scale * jacobian_position)
            weighted_errors.append(weight_scale * error)
            raw_errors.append(error)
        return (
            np.vstack(jacobian_blocks),
            np.hstack(weighted_errors),
            np.hstack(raw_errors),
        )

    def _solve_task_dls(
        self,
        jacobian: FloatArray,
        error: FloatArray,
        damping: FloatArray,
    ) -> FloatArray:
        if self.reference_ordered:
            return _solve_dls_reference(jacobian, error, damping)
        return _solve_dls(jacobian, error, damping)

    def _error_norm(self, error: FloatArray) -> float:
        if not self.reference_ordered:
            return float(np.linalg.norm(error))
        squared = 0.0
        for value in error:
            component = float(value)
            squared += component * component
        return math.sqrt(squared)

    def _free_base_dof_indices(self) -> NDArray[np.int32] | None:
        for joint_id in range(int(self.model.njnt)):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                start = int(self.model.jnt_dofadr[joint_id])
                return np.arange(start, start + 6, dtype=np.int32)
        return None

    def _apply_base_update(
        self,
        jacobian: FloatArray,
        error: FloatArray,
        dof_indices: NDArray[np.int32],
        *,
        step: float,
        position_limit: float,
        rotation_limit: float,
    ) -> None:
        base_jacobian = jacobian[:, dof_indices]
        base_damping = np.full(6, max(float(np.min(self._damping)), 1e-12))
        delta = step * self._solve_task_dls(base_jacobian, error, base_damping)
        if not np.isfinite(delta).all():
            delta = np.zeros(6, dtype=np.float64)
        delta = np.hstack(
            (_trim_scale(delta[:3], position_limit), _trim_scale(delta[3:], rotation_limit))
        )
        velocity = np.zeros(self.model.nv, dtype=np.float64)
        velocity[dof_indices] = delta
        mujoco.mj_integratePos(self.model, self.data.qpos, velocity, 1.0)
        mujoco.mj_forward(self.model, self.data)

    def _joint_delta(
        self,
        jacobian: FloatArray,
        error: FloatArray,
        use_indices: NDArray[np.int64],
        *,
        joint_limits: bool,
        nullspace: bool,
    ) -> FloatArray:
        joint_jacobian = jacobian[:, self._dof_indices]
        selected_jacobian = joint_jacobian[:, use_indices]
        damping = self._damping[use_indices]
        is_revolute = self._is_revolute[use_indices]
        steps = np.where(is_revolute, self.revolute_step, self.prismatic_step)
        selected_delta = self._solve_task_dls(selected_jacobian, error, damping) * steps
        if not np.isfinite(selected_delta).all():
            selected_delta = np.zeros_like(selected_delta)

        valid_mask = np.ones(len(use_indices), dtype=np.bool_)
        current_qpos = np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64)
        if joint_limits:
            probes = np.where(is_revolute, self.joint_limit_probe, self.prismatic_limit_probe)
            selected_qpos = current_qpos[use_indices]
            probe_qpos = selected_qpos + probes * np.sign(selected_delta)
            minimum = self._minimum[use_indices]
            maximum = self._maximum[use_indices]
            violates = ((probe_qpos > maximum) & (selected_delta > 0.0)) | (
                (probe_qpos < minimum) & (selected_delta < 0.0)
            )
            valid_mask = ~violates
            if np.any(violates):
                valid_local = np.flatnonzero(valid_mask).astype(np.int64)
                selected_delta = np.zeros_like(selected_delta)
                if len(valid_local):
                    reduced_jacobian = selected_jacobian[:, valid_local]
                    reduced = self._solve_task_dls(
                        reduced_jacobian,
                        error,
                        damping[valid_local],
                    )
                    selected_delta[valid_local] = reduced * steps[valid_local]

        valid_local = np.flatnonzero(valid_mask).astype(np.int64)
        if nullspace and self._home is not None and len(valid_local):
            valid_joint_indices = use_indices[valid_local]
            valid_jacobian = selected_jacobian[:, valid_local]
            null_delta = self.nullspace_gain * (
                self._home[valid_joint_indices] - current_qpos[valid_joint_indices]
            )
            if self.reference_ordered:
                null_delta = _mask_task_irrelevant_reference(valid_jacobian, null_delta)
                null_delta = _project_nullspace_reference(
                    valid_jacobian,
                    null_delta,
                    damping[valid_local],
                )
            else:
                null_delta = _mask_task_irrelevant(valid_jacobian, null_delta)
                null_delta = _project_nullspace(
                    valid_jacobian,
                    null_delta,
                    damping[valid_local],
                )
            selected_delta[valid_local] += null_delta * steps[valid_local]

        result = np.zeros(len(self.joint_names), dtype=np.float64)
        result[use_indices] = selected_delta
        if not np.isfinite(result).all():
            return np.zeros_like(result)
        return result

    def _apply_joint_update(self, delta: FloatArray) -> None:
        current = np.asarray(self.data.qpos[self._qpos_indices], dtype=np.float64)
        result = current.copy()
        revolute_delta = _trim_scale(delta[self._is_revolute], self.revolute_update_limit)
        prismatic_delta = _trim_scale(delta[~self._is_revolute], self.prismatic_update_limit)
        result[self._is_revolute] += revolute_delta
        result[~self._is_revolute] += prismatic_delta
        result = np.clip(result, self._minimum, self._maximum)
        self.data.qpos[self._qpos_indices] = result
        mujoco.mj_forward(self.model, self.data)

    def _solve_native(
        self,
        selected_names: tuple[str, ...],
        use_indices: NDArray[np.int64],
        *,
        joint_limits: bool,
        nullspace: bool,
        base_control: bool,
        base_step: float,
        base_position_limit: float,
        base_rotation_limit: float,
    ) -> IkResult:
        """Run the nanobind loop with this solver's validated state."""

        model_address, data_address = mujoco_addresses(self.model, self.data)
        body_ids = np.ascontiguousarray(
            [target.body_id for target in self._targets],
            dtype=np.int32,
        )
        local_positions = np.ascontiguousarray(
            np.vstack([target.local_position for target in self._targets]),
            dtype=np.float64,
        )
        target_positions = np.ascontiguousarray(
            np.vstack([target.target_position for target in self._targets]),
            dtype=np.float64,
        )
        weights = np.ascontiguousarray(
            [target.weight for target in self._targets],
            dtype=np.float64,
        )
        use_indices_i32 = np.ascontiguousarray(use_indices, dtype=np.int32)
        active_indices = np.arange(len(self.joint_names), dtype=np.int32)
        home = (
            np.empty(0, dtype=np.float64)
            if self._home is None
            else np.ascontiguousarray(self._home, dtype=np.float64)
        )
        empty_indices = np.empty(0, dtype=np.int32)
        empty_coefficients = np.empty((0, 5), dtype=np.float64)
        empty_values = np.empty(0, dtype=np.float64)
        base_dof_indices = self._free_base_dof_indices() if base_control else None
        base_control_use = base_dof_indices is not None
        base_indices = (
            np.ascontiguousarray(base_dof_indices, dtype=np.int32)
            if base_control_use
            else np.empty(0, dtype=np.int32)
        )

        arrays = self.backend.module.solve_body_position_ik(
            model_address,
            data_address,
            body_ids,
            local_positions,
            target_positions,
            weights,
            np.ascontiguousarray(self._qpos_indices, dtype=np.int32),
            np.ascontiguousarray(self._dof_indices, dtype=np.int32),
            use_indices_i32,
            active_indices,
            np.ascontiguousarray(self._is_revolute, dtype=np.int32),
            np.ascontiguousarray(self._minimum, dtype=np.float64),
            np.ascontiguousarray(self._maximum, dtype=np.float64),
            home,
            np.ascontiguousarray(self._damping[use_indices], dtype=np.float64),
            empty_indices,
            empty_indices,
            empty_coefficients,
            empty_values,
            empty_values,
            base_indices,
            bool(base_control_use),
            max(float(np.min(self._damping)), 1e-12),
            float(base_step),
            float(base_position_limit),
            float(base_rotation_limit),
            int(self.max_iterations),
            float(self.revolute_step),
            float(self.prismatic_step),
            float(self.revolute_update_limit),
            float(self.prismatic_update_limit),
            float(self.joint_limit_probe),
            float(self.prismatic_limit_probe),
            float(self.nullspace_gain),
            bool(joint_limits),
            bool(nullspace),
            True,
        )
        qpos = np.asarray(arrays["qpos_full_best"], dtype=np.float64).copy()
        errors = np.asarray(arrays["ik_err_array"], dtype=np.float64).copy()
        best_iteration = int(arrays["idx_best"])
        return IkResult(
            qpos=qpos,
            joint_qpos=np.asarray(arrays["q_rev_pri_best"], dtype=np.float64).copy(),
            error=float(arrays["ik_err_best"]),
            iterations=self.max_iterations,
            errors=errors,
            best_iteration=best_iteration,
            joints=selected_names,
        )

    def solve(
        self,
        source_model: MuJoCoModelAdapter | None = None,
        joints: Sequence[str] | None = None,
        joint_limits: bool = True,
        nullspace: bool = False,
        base_control: bool = False,
        base_step: float = 0.5,
        base_position_limit: float = 0.1,
        base_rotation_limit: float = np.deg2rad(5.0),
    ) -> IkResult:
        """Solve buffered targets and return the best candidate pose."""

        if source_model is not None:
            self._sync_from(source_model)
        for name, value in {
            "base_step": base_step,
            "base_position_limit": base_position_limit,
            "base_rotation_limit": base_rotation_limit,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        selected_names = self.joint_names if joints is None else tuple(joints)
        unknown = set(selected_names) - set(self.joint_names)
        if unknown:
            raise ValueError(f"unknown IK joints: {sorted(unknown)}")
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("joints contains duplicates")
        name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        use_indices = np.asarray([name_to_index[name] for name in selected_names], dtype=np.int64)

        if not self._targets:
            qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
            return IkResult(
                qpos=qpos,
                joint_qpos=qpos[self._qpos_indices].copy(),
                error=0.0,
                iterations=0,
                errors=np.asarray([0.0], dtype=np.float64),
                best_iteration=0,
                joints=selected_names,
            )

        if self.backend.is_native and len(use_indices):
            return self._solve_native(
                selected_names,
                use_indices,
                joint_limits=joint_limits,
                nullspace=nullspace,
                base_control=base_control,
                base_step=base_step,
                base_position_limit=base_position_limit,
                base_rotation_limit=base_rotation_limit,
            )

        _, _, raw_error = self._build_weighted_system()
        errors = np.empty(self.max_iterations + 1, dtype=np.float64)
        errors[0] = self._error_norm(raw_error)
        candidates = [np.asarray(self.data.qpos, dtype=np.float64).copy()]
        base_dof_indices = self._free_base_dof_indices() if base_control else None

        for iteration in range(self.max_iterations):
            jacobian, weighted_error, _ = self._build_weighted_system()
            if base_dof_indices is not None:
                self._apply_base_update(
                    jacobian,
                    weighted_error,
                    base_dof_indices,
                    step=base_step,
                    position_limit=base_position_limit,
                    rotation_limit=base_rotation_limit,
                )
                jacobian, weighted_error, _ = self._build_weighted_system()

            delta = self._joint_delta(
                jacobian,
                weighted_error,
                use_indices,
                joint_limits=joint_limits,
                nullspace=nullspace,
            )
            self._apply_joint_update(delta)
            _, _, raw_error = self._build_weighted_system()
            candidates.append(np.asarray(self.data.qpos, dtype=np.float64).copy())
            errors[iteration + 1] = self._error_norm(raw_error)

        best_iteration = int(np.argmin(errors))
        qpos = candidates[best_iteration]
        return IkResult(
            qpos=qpos,
            joint_qpos=qpos[self._qpos_indices].copy(),
            error=float(errors[best_iteration]),
            iterations=self.max_iterations,
            errors=errors,
            best_iteration=best_iteration,
            joints=selected_names,
        )


__all__ = ["BodyPositionIKSolver", "IkResult", "MuJoCoModelAdapter"]
