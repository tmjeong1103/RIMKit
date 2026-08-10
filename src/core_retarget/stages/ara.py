"""Contact-aware affine root adjustment (ARA).

Stage 5 first applies one global vertical grounding offset derived from the
lowest contact-segment median toe height.  It then solves a bounded affine
scale/shift problem for root XY motion and applies the same root displacement
to both ankles and toes.  It does not materialize a new qpos trajectory; that
happens when Stage 6 constructs the FPA targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.motion.contacts import ContactSchedule
from core_retarget.robots.profiles.ara import AraProfile, get_ara_profile
from core_retarget.stages.target_trajectories import TargetTrajectoriesResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_SUCCESS_STATUSES = frozenset(("optimal", "optimal_inaccurate"))
_SOLVER = "CLARABEL"


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AraSlipStats:
    """Toe XY motion within the source contact segments."""

    step_max: float
    step_mean: float
    segment_spread_max: float


@dataclass(frozen=True, slots=True)
class AraFloorStats:
    """Absolute toe-height error against the shared ARA floor target."""

    max_abs_error: float
    mean_abs_error: float


@dataclass(frozen=True, slots=True)
class AraDiagnostics:
    """Typed solver and motion diagnostics emitted by Stage 5."""

    backend: str
    solver: str
    status: str
    objective: float
    lowest_contact_z: float
    ground_translation_z: float
    right_slip_pre: AraSlipStats
    right_slip_post: AraSlipStats
    left_slip_pre: AraSlipStats
    left_slip_post: AraSlipStats
    right_floor_z: AraFloorStats
    left_floor_z: AraFloorStats
    root_xy_bound_violation: float


@dataclass(frozen=True, slots=True)
class AraResult:
    """Immutable Stage 5 grounded and affine-adjusted landmark trajectories."""

    robot_id: str
    fps: float
    seconds: FloatArray
    ground_offset: FloatArray
    scale: FloatArray
    shift: FloatArray
    root_grounded: FloatArray
    right_ankle_grounded: FloatArray
    left_ankle_grounded: FloatArray
    right_toe_grounded: FloatArray
    left_toe_grounded: FloatArray
    root_ara: FloatArray
    right_ankle_ara: FloatArray
    left_ankle_ara: FloatArray
    right_toe_ara: FloatArray
    left_toe_ara: FloatArray
    root_xy_shift: FloatArray
    root_xy_expected_bound: FloatArray
    toe_floor_target_z: float
    diagnostics: AraDiagnostics

    def __post_init__(self) -> None:
        if not str(self.robot_id).strip():
            raise ValueError("ARA robot_id must not be empty.")
        try:
            fps = float(self.fps)
            floor_z = float(self.toe_floor_target_z)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ARA scalar metadata must be finite.") from exc
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("ARA fps must be finite and positive.")
        if not np.isfinite(floor_z):
            raise ValueError("ARA toe floor target must be finite.")
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "toe_floor_target_z", floor_z)

        seconds = _readonly_float(self.seconds).reshape(-1)
        if seconds.size == 0 or not np.isfinite(seconds).all():
            raise ValueError("ARA seconds must be a non-empty finite vector.")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("ARA seconds must be strictly increasing.")
        object.__setattr__(self, "seconds", seconds)

        fixed_shapes = {
            "ground_offset": (1, 3),
            "scale": (1, 3),
            "shift": (1, 3),
            "root_xy_shift": (len(seconds), 2),
            "root_xy_expected_bound": (len(seconds), 2),
        }
        trajectory_fields = (
            "root_grounded",
            "right_ankle_grounded",
            "left_ankle_grounded",
            "right_toe_grounded",
            "left_toe_grounded",
            "root_ara",
            "right_ankle_ara",
            "left_ankle_ara",
            "right_toe_ara",
            "left_toe_ara",
        )
        fixed_shapes.update({name: (len(seconds), 3) for name in trajectory_fields})
        for field_name, shape in fixed_shapes.items():
            value = _readonly_float(getattr(self, field_name))
            if value.shape != shape:
                raise ValueError(f"ARA field {field_name} must have shape {shape}.")
            if not np.isfinite(value).all():
                raise ValueError(f"ARA field {field_name} contains NaN or infinity.")
            object.__setattr__(self, field_name, value)

    @property
    def frame_count(self) -> int:
        """Number of trajectory frames."""

        return int(self.seconds.shape[0])

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 5 archive arrays."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "ground_offset": self.ground_offset,
            "s_val": self.scale,
            "b_val": self.shift,
            "p_root_trgt_grd_array": self.root_grounded,
            "p_ra_trgt_grd_array": self.right_ankle_grounded,
            "p_la_trgt_grd_array": self.left_ankle_grounded,
            "p_rt_trgt_grd_array": self.right_toe_grounded,
            "p_lt_trgt_grd_array": self.left_toe_grounded,
            "p_root_trgt_ara_array": self.root_ara,
            "p_ra_trgt_ara_array": self.right_ankle_ara,
            "p_la_trgt_ara_array": self.left_ankle_ara,
            "p_rt_trgt_ara_array": self.right_toe_ara,
            "p_lt_trgt_ara_array": self.left_toe_ara,
            "root_xy_shift": self.root_xy_shift,
            "root_xy_expected_bound": self.root_xy_expected_bound,
        }


def _expand_segments(bounds: NDArray[np.integer[Any]], frame_count: int) -> tuple[IntArray, ...]:
    values = np.asarray(bounds, dtype=np.int64)
    if values.ndim != 2 or values.shape[1:] != (2,):
        raise MotionValidationError("ARA contact segments must have shape (N, 2).")
    if np.any(values < 0) or np.any(values[:, 0] >= values[:, 1]):
        raise MotionValidationError("ARA contact segments contain invalid bounds.")
    if np.any(values[:, 1] > frame_count):
        raise MotionValidationError("ARA contact segments exceed the frame count.")
    return tuple(np.arange(int(start), int(stop), dtype=np.int64) for start, stop in values)


def _validate_inputs(
    targets: TargetTrajectoriesResult,
    contacts: ContactSchedule,
    robot_id: str,
) -> tuple[AraProfile, tuple[IntArray, ...], tuple[IntArray, ...]]:
    profile = get_ara_profile(robot_id)
    if targets.robot_id != robot_id:
        raise ConfigurationError(
            "Target-trajectory robot and ARA robot must match "
            f"({targets.robot_id!r} != {robot_id!r})."
        )
    if targets.frame_count != contacts.frame_count:
        raise MotionValidationError(
            "Stage 4 trajectories and contact schedule must have the same frame count."
        )
    if not np.array_equal(targets.seconds, contacts.seconds):
        raise MotionValidationError(
            "Stage 4 trajectories and contact schedule must share the same timestamps."
        )
    if not np.isclose(targets.fps, contacts.fps, rtol=0.0, atol=1e-12):
        raise MotionValidationError(
            "Stage 4 trajectories and contact schedule must share the same FPS."
        )
    right = _expand_segments(contacts.right_contact_segments, targets.frame_count)
    left = _expand_segments(contacts.left_contact_segments, targets.frame_count)
    return profile, right, left


def _contact_segment_z_candidates(
    segments: tuple[IntArray, ...],
    toe: FloatArray,
) -> list[float]:
    return [float(np.median(toe[segment, 2])) for segment in segments if len(segment)]


def _contact_terms(
    segments: tuple[IntArray, ...],
    affine_toe: Any,
    *,
    frame_count: int,
    target_z: float,
    profile: AraProfile,
    name_prefix: str,
) -> tuple[list[Any], list[Any]]:
    first_difference = np.zeros((frame_count - 1, frame_count), dtype=np.float64)
    for index in range(frame_count - 1):
        first_difference[index, index] = -1.0
        first_difference[index, index + 1] = 1.0

    objectives: list[Any] = []
    constraints: list[Any] = []
    for segment in segments:
        first_tick, last_tick = int(segment[0]), int(segment[-1])
        if last_tick <= first_tick:
            continue
        segment_difference = first_difference[np.arange(first_tick, last_tick)]
        x_slack = cp.Variable(
            last_tick - first_tick,
            name=f"{name_prefix}_x_slack_{first_tick}_{last_tick}",
        )
        y_slack = cp.Variable(
            last_tick - first_tick,
            name=f"{name_prefix}_y_slack_{first_tick}_{last_tick}",
        )
        constraints.extend(
            (
                segment_difference @ affine_toe[:, 0] == x_slack,
                segment_difference @ affine_toe[:, 1] == y_slack,
            )
        )
        objectives.append(
            profile.slip_weight
            * (
                cp.sum_squares(x_slack)  # type: ignore[attr-defined]
                + cp.sum_squares(y_slack)  # type: ignore[attr-defined]
            )
        )

        segment_length = last_tick - first_tick + 1
        selector = np.zeros((segment_length, frame_count), dtype=np.float64)
        selector[np.arange(segment_length), np.arange(first_tick, last_tick + 1)] = 1.0
        z_slack = cp.Variable(
            segment_length,
            name=f"{name_prefix}_z_slack_{first_tick}_{last_tick}",
        )
        constraints.append(selector @ affine_toe[:, 2] - float(target_z) == z_slack)
        objectives.append(
            profile.floor_lock_weight * cp.sum_squares(z_slack)  # type: ignore[attr-defined]
        )
    return objectives, constraints


def _solve_affine_adjustment(
    root_grounded: FloatArray,
    right_toe_grounded: FloatArray,
    left_toe_grounded: FloatArray,
    right_segments: tuple[IntArray, ...],
    left_segments: tuple[IntArray, ...],
    target_z: float,
    profile: AraProfile,
) -> tuple[FloatArray, FloatArray, str, float]:
    scale_variable = cp.Variable(3)
    shift_variable = cp.Variable(3)
    scale_row = cp.reshape(  # type: ignore[attr-defined]
        scale_variable, (1, 3), order="C"
    )
    shift_row = cp.reshape(  # type: ignore[attr-defined]
        shift_variable, (1, 3), order="C"
    )
    right_toe_constant = cp.Constant(right_toe_grounded)
    left_toe_constant = cp.Constant(left_toe_grounded)
    root_constant = cp.Constant(root_grounded)
    root_affine = (
        cp.multiply(  # type: ignore[attr-defined]
            root_constant, scale_row
        )
        + shift_row
    )
    root_difference = root_affine - root_constant
    right_toe_affine = right_toe_constant + root_difference
    left_toe_affine = left_toe_constant + root_difference

    objectives: list[Any] = [
        profile.scale_regularization_weight * cp.sum_squares(scale_variable - 1.0),  # type: ignore[attr-defined]
        profile.shift_regularization_weight * cp.sum_squares(shift_variable[:2]),  # type: ignore[attr-defined]
    ]
    constraints: list[Any] = [
        scale_variable[2] == 1.0,
        shift_variable[2] == 0.0,
        scale_variable[0] >= 1.0 - profile.scale_xy_bound,
        scale_variable[0] <= 1.0 + profile.scale_xy_bound,
        scale_variable[1] >= 1.0 - profile.scale_xy_bound,
        scale_variable[1] <= 1.0 + profile.scale_xy_bound,
        shift_variable[0] >= -profile.shift_xy_bound,
        shift_variable[0] <= profile.shift_xy_bound,
        shift_variable[1] >= -profile.shift_xy_bound,
        shift_variable[1] <= profile.shift_xy_bound,
    ]
    right_objectives, right_constraints = _contact_terms(
        right_segments,
        right_toe_affine,
        frame_count=len(root_grounded),
        target_z=target_z,
        profile=profile,
        name_prefix="rt",
    )
    left_objectives, left_constraints = _contact_terms(
        left_segments,
        left_toe_affine,
        frame_count=len(root_grounded),
        target_z=target_z,
        profile=profile,
        name_prefix="lt",
    )
    objectives.extend(right_objectives + left_objectives)
    constraints.extend(right_constraints + left_constraints)

    problem = cp.Problem(
        cp.Minimize(cp.sum(objectives)),  # type: ignore[attr-defined]
        constraints,
    )
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise RuntimeError("ARA optimization failed while running CLARABEL.") from exc
    status = str(problem.status)
    if status not in _SUCCESS_STATUSES:
        raise RuntimeError(f"ARA optimization did not converge (status={status}).")
    if scale_variable.value is None or shift_variable.value is None or problem.value is None:
        raise RuntimeError("ARA optimization returned no solution.")

    scale = np.asarray(scale_variable.value, dtype=np.float64).reshape(1, 3)
    shift = np.asarray(shift_variable.value, dtype=np.float64).reshape(1, 3)
    objective = float(problem.value)
    if not np.isfinite(scale).all() or not np.isfinite(shift).all():
        raise RuntimeError("ARA optimization returned a non-finite solution.")
    if not np.isfinite(objective):
        raise RuntimeError("ARA optimization returned a non-finite objective.")
    if np.any(np.abs(scale[0, :2] - 1.0) > profile.scale_xy_bound + 1e-6) or np.any(
        np.abs(shift[0, :2]) > profile.shift_xy_bound + 1e-6
    ):
        raise RuntimeError(
            f"ARA optimization returned values outside XY bounds: s={scale} b={shift}."
        )
    return scale, shift, status, objective


def _contact_xy_slip_stats(
    positions: FloatArray,
    segments: tuple[IntArray, ...],
) -> AraSlipStats:
    step_slips: list[float] = []
    segment_spreads: list[float] = []
    for segment in segments:
        if len(segment) <= 1:
            continue
        xy = positions[segment, :2]
        step_slips.extend(np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist())
        segment_spreads.append(float(np.max(np.linalg.norm(xy - xy[0], axis=1))))
    if not step_slips:
        return AraSlipStats(0.0, 0.0, 0.0)
    return AraSlipStats(
        step_max=float(np.max(step_slips)),
        step_mean=float(np.mean(step_slips)),
        segment_spread_max=float(np.max(segment_spreads)),
    )


def _contact_floor_stats(
    positions: FloatArray,
    segments: tuple[IntArray, ...],
    target_z: float,
) -> AraFloorStats:
    errors: list[float] = []
    for segment in segments:
        if len(segment):
            errors.extend(np.abs(positions[segment, 2] - float(target_z)).tolist())
    if not errors:
        return AraFloorStats(0.0, 0.0)
    return AraFloorStats(
        max_abs_error=float(np.max(errors)),
        mean_abs_error=float(np.mean(errors)),
    )


def run_ara(
    targets: TargetTrajectoriesResult,
    contacts: ContactSchedule,
    *,
    robot_id: str,
) -> AraResult:
    """Run the shared Stage 5 ARA solve."""

    profile, right_segments, left_segments = _validate_inputs(targets, contacts, robot_id)
    contact_z_candidates = _contact_segment_z_candidates(
        right_segments, targets.right_toe_smoothed
    ) + _contact_segment_z_candidates(left_segments, targets.left_toe_smoothed)
    if contact_z_candidates:
        lowest_contact_z = float(np.min(contact_z_candidates))
    else:
        lowest_contact_z = float(
            min(
                np.min(targets.right_toe_smoothed[:, 2]),
                np.min(targets.left_toe_smoothed[:, 2]),
            )
        )
    ground_translation_z = float(profile.ground_target_z - lowest_contact_z)
    ground_offset = np.array([[0.0, 0.0, ground_translation_z]], dtype=np.float64)

    root_grounded = targets.root_smoothed + ground_offset
    right_ankle_grounded = targets.right_ankle_smoothed + ground_offset
    left_ankle_grounded = targets.left_ankle_smoothed + ground_offset
    right_toe_grounded = targets.right_toe_smoothed + ground_offset
    left_toe_grounded = targets.left_toe_smoothed + ground_offset
    toe_floor_target_z = float(profile.ground_target_z + profile.toe_floor_offset)

    scale, shift, status, objective = _solve_affine_adjustment(
        root_grounded,
        right_toe_grounded,
        left_toe_grounded,
        right_segments,
        left_segments,
        toe_floor_target_z,
        profile,
    )
    root_ara = root_grounded * scale + shift
    root_difference = root_ara - root_grounded
    right_toe_ara = right_toe_grounded + root_difference
    left_toe_ara = left_toe_grounded + root_difference
    right_ankle_ara = right_ankle_grounded + root_difference
    left_ankle_ara = left_ankle_grounded + root_difference

    root_xy_shift = root_ara[:, :2] - root_grounded[:, :2]
    root_xy_expected_bound = (
        np.abs(root_grounded[:, :2]) * profile.scale_xy_bound + profile.shift_xy_bound + 1e-6
    )
    root_xy_bound_violation = float(np.max(np.abs(root_xy_shift) - root_xy_expected_bound))
    if root_xy_bound_violation > 1e-5:
        raise RuntimeError("ARA root XY bound was violated.")

    diagnostics = AraDiagnostics(
        backend="cvxpy",
        solver=_SOLVER,
        status=status,
        objective=objective,
        lowest_contact_z=lowest_contact_z,
        ground_translation_z=ground_translation_z,
        right_slip_pre=_contact_xy_slip_stats(right_toe_grounded, right_segments),
        right_slip_post=_contact_xy_slip_stats(right_toe_ara, right_segments),
        left_slip_pre=_contact_xy_slip_stats(left_toe_grounded, left_segments),
        left_slip_post=_contact_xy_slip_stats(left_toe_ara, left_segments),
        right_floor_z=_contact_floor_stats(right_toe_ara, right_segments, toe_floor_target_z),
        left_floor_z=_contact_floor_stats(left_toe_ara, left_segments, toe_floor_target_z),
        root_xy_bound_violation=root_xy_bound_violation,
    )
    return AraResult(
        robot_id=robot_id,
        fps=targets.fps,
        seconds=targets.seconds,
        ground_offset=ground_offset,
        scale=scale,
        shift=shift,
        root_grounded=root_grounded,
        right_ankle_grounded=right_ankle_grounded,
        left_ankle_grounded=left_ankle_grounded,
        right_toe_grounded=right_toe_grounded,
        left_toe_grounded=left_toe_grounded,
        root_ara=root_ara,
        right_ankle_ara=right_ankle_ara,
        left_ankle_ara=left_ankle_ara,
        right_toe_ara=right_toe_ara,
        left_toe_ara=left_toe_ara,
        root_xy_shift=root_xy_shift,
        root_xy_expected_bound=root_xy_expected_bound,
        toe_floor_target_z=toe_floor_target_z,
        diagnostics=diagnostics,
    )


__all__ = [
    "AraDiagnostics",
    "AraFloorStats",
    "AraResult",
    "AraSlipStats",
    "run_ara",
]
