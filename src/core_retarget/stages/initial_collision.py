"""First post-DMR signed-distance self-collision refinement.

The stage applies configured solve passes, margins, arm-only smoothing, and a
final unsmoothed pass without stance normalization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.mujoco.collision import (
    CollisionCandidateSet,
    SignedDistanceBatch,
    build_collision_candidates,
    query_signed_distances,
)
from core_retarget.mujoco.ik import BodyPositionIKSolver
from core_retarget.mujoco.model import MujocoModel
from core_retarget.native import BackendPreference, BackendSelection, resolve_backend
from core_retarget.optimization import shape_trajectory_1d
from core_retarget.robots.profiles import (
    InitialCollisionProfile,
    get_initial_collision_profile,
)

FloatArray = NDArray[np.float64]
InitialCollisionProgress = Callable[[int, int, int, float], None]
_REFERENCE_MUJOCO_VERSION = "3.6.0"


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CollisionPassDiagnostics:
    """Violation count measured after one outer pass at that pass's margin."""

    margin: float
    violations: int
    max_frame_violations: int


@dataclass(frozen=True, slots=True)
class InitialCollisionDiagnostics:
    """Structured runtime facts from initial collision refinement.

    ``trajectory_backend`` identifies the configured CVXPY solver/fallback
    policy. It is not a per-joint solver trace; the stage
    retains the unsmoothed joint trajectory if every shaping attempt fails.
    """

    backend: str
    distance_backend: str
    ik_backend: str
    trajectory_backend: str
    root_geom_count: int
    collision_geom_count: int
    raw_candidate_pair_count: int
    candidate_pair_count: int
    arm_joint_names: tuple[str, ...]
    input_violations: int
    input_max_frame_violations: int
    output_violations: int
    output_max_frame_violations: int
    passes: tuple[CollisionPassDiagnostics, ...]

    @property
    def outer_passes(self) -> int:
        return len(self.passes)


@dataclass(frozen=True, slots=True)
class InitialCollisionResult:
    """Stage 3 trajectory and diagnostics before later CoRe stages."""

    robot_id: str
    fps: float
    seconds: FloatArray
    qpos: FloatArray
    diagnostics: InitialCollisionDiagnostics

    def __post_init__(self) -> None:
        seconds = _readonly_float(self.seconds).reshape(-1)
        qpos = _readonly_float(self.qpos)
        if qpos.ndim != 2 or qpos.shape[0] != len(seconds):
            raise ValueError("initial-collision qpos must have shape (frames, nq)")
        if not np.isfinite(seconds).all() or not np.isfinite(qpos).all():
            raise ValueError("initial-collision arrays must contain only finite values")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("initial-collision fps must be finite and positive")
        seconds.setflags(write=False)
        qpos.setflags(write=False)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "qpos", qpos)

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the exact four-array Stage 3 archive contract."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_cc_smt_array": self.qpos,
        }


@dataclass(frozen=True, slots=True)
class _DepenetrationTarget:
    body_id: int
    current: FloatArray
    target: FloatArray
    weight: float
    violation: float


def _validate_reference_environment() -> None:
    version = str(getattr(mujoco, "__version__", "unknown"))
    if version != _REFERENCE_MUJOCO_VERSION:
        raise ConfigurationError(
            "Initial collision refinement is numerically qualified with MuJoCo "
            f"{_REFERENCE_MUJOCO_VERSION}; the loaded version is {version}. "
            "Install the version required by core-retarget before running Stage 3."
        )


def _validate_inputs(
    qpos: ArrayLike,
    seconds: ArrayLike,
    fps: float,
    profile: InitialCollisionProfile,
) -> tuple[FloatArray, FloatArray]:
    trajectory = np.asarray(qpos, dtype=np.float64)
    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if trajectory.ndim != 2 or trajectory.shape != (len(time_values), profile.qpos_dim):
        raise MotionValidationError(
            f"Stage 3 qpos must have shape (frames, {profile.qpos_dim}); found {trajectory.shape}."
        )
    if len(time_values) == 0:
        raise MotionValidationError("Stage 3 input must contain at least one frame.")
    if not np.isfinite(trajectory).all() or not np.isfinite(time_values).all():
        raise MotionValidationError("Stage 3 input contains NaN or infinity.")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise MotionValidationError("Stage 3 seconds must be strictly increasing.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise MotionValidationError("Stage 3 fps must be finite and positive.")
    return trajectory.copy(order="C"), time_values.copy(order="C")


def _violation_batch(
    model: MujocoModel,
    candidates: CollisionCandidateSet,
    margin: float,
    profile: InitialCollisionProfile,
    backend: BackendSelection,
) -> tuple[SignedDistanceBatch, NDArray[np.int64]]:
    batch = query_signed_distances(
        model,
        candidates.geom_pairs,
        limit=profile.query_limit,
        backend=backend,
    )
    indices = np.flatnonzero(batch.distance < margin).astype(np.int64)
    return batch, indices


def _depenetration_targets(
    batch: SignedDistanceBatch,
    indices: NDArray[np.int64],
    candidates: CollisionCandidateSet,
    margin: float,
    profile: InitialCollisionProfile,
) -> list[_DepenetrationTarget]:
    targets: list[_DepenetrationTarget] = []
    for index_value in indices:
        index = int(index_value)
        signed_distance = float(batch.distance[index])
        body1 = int(batch.body_pairs[index, 0])
        body2 = int(batch.body_pairs[index, 1])
        body1_movable = body1 in candidates.movable_body_ids
        body2_movable = body2 in candidates.movable_body_ids
        if not body1_movable and not body2_movable:
            continue

        point1 = np.asarray(batch.fromto[index, :3], dtype=np.float64).copy()
        point2 = np.asarray(batch.fromto[index, 3:], dtype=np.float64).copy()
        if int(batch.source[index]) == 1:
            point = 0.5 * (point1 + point2)
            point1 = point.copy()
            point2 = point.copy()
            push_direction = np.asarray(batch.normal[index], dtype=np.float64).copy()
        else:
            push_direction = point2 - point1
        push_norm = float(np.linalg.norm(push_direction))
        if push_norm <= 1e-9 or not np.isfinite(push_norm):
            continue
        push_direction /= push_norm
        violation = margin - signed_distance
        step = min(profile.correction_gain * violation, profile.correction_length_cap)
        weight_alpha = float(np.clip(violation / max(margin, 1e-6), 0.0, 1.0))
        weight = 0.2 + 1.8 * weight_alpha
        if body1_movable:
            targets.append(
                _DepenetrationTarget(
                    body_id=body1,
                    current=point1,
                    target=point1 - step * push_direction,
                    weight=weight,
                    violation=violation,
                )
            )
        if body2_movable:
            targets.append(
                _DepenetrationTarget(
                    body_id=body2,
                    current=point2,
                    target=point2 + step * push_direction,
                    weight=weight,
                    violation=violation,
                )
            )
    targets.sort(key=lambda target: target.violation, reverse=True)
    return targets


def _add_orientation_targets(
    visible_model: MujocoModel,
    solver: BodyPositionIKSolver,
    targets: Sequence[_DepenetrationTarget],
    profile: InitialCollisionProfile,
) -> None:
    if not profile.preserve_orientation or profile.orientation_weight <= 0.0:
        return
    displacement_by_body: dict[int, tuple[FloatArray, float]] = {}
    for target in targets[: profile.target_limit]:
        displacement = target.target - target.current
        if not np.isfinite(displacement).all() or target.weight <= 0.0:
            continue
        total, total_weight = displacement_by_body.get(
            target.body_id,
            (np.zeros(3, dtype=np.float64), 0.0),
        )
        displacement_by_body[target.body_id] = (
            total + target.weight * displacement,
            total_weight + target.weight,
        )

    for body_id, (total, total_weight) in displacement_by_body.items():
        if total_weight <= 1e-12:
            continue
        body_name = visible_model.body_names[body_id]
        current_transform = visible_model.get_body_transform(body_name)
        target_transform = current_transform.copy()
        target_transform[:3, 3] += total / total_weight
        solver.add_transform_target(
            body_name,
            current_transform,
            target_transform,
            weight=profile.orientation_weight,
            axis_length=profile.orientation_axis_length,
        )


def _collision_pass(
    visible_model: MujocoModel,
    solver: BodyPositionIKSolver,
    qpos: FloatArray,
    candidates: CollisionCandidateSet,
    profile: InitialCollisionProfile,
    margin: float,
    outer_pass: int,
    progress: InitialCollisionProgress | None,
    backend: BackendSelection,
) -> FloatArray:
    output = np.empty_like(qpos, dtype=np.float64)
    frame_count = len(qpos)
    for frame_index, frame_qpos in enumerate(qpos):
        visible_model.forward(frame_qpos)
        for _ in range(profile.ticks_per_pass):
            batch, violation_indices = _violation_batch(
                visible_model,
                candidates,
                margin,
                profile,
                backend,
            )
            if len(violation_indices) == 0:
                break
            targets = _depenetration_targets(
                batch,
                violation_indices,
                candidates,
                margin,
                profile,
            )
            if not targets:
                break

            solver.reset_targets(sync_from=visible_model)
            for target in targets[: profile.target_limit]:
                solver.add_target(
                    visible_model.body_names[target.body_id],
                    target.current,
                    target.target,
                    target.weight,
                )
            _add_orientation_targets(visible_model, solver, targets, profile)
            result = solver.solve(
                source_model=visible_model,
                joints=candidates.arm_joint_names,
                joint_limits=True,
                nullspace=False,
                base_control=False,
            )
            visible_model.forward(result.qpos)
        output[frame_index] = visible_model.get_qpos()
        if progress is not None:
            progress(outer_pass + 1, frame_index + 1, frame_count, margin)
    return output


def _smooth_arm_joints(
    model: MujocoModel,
    qpos: FloatArray,
    seconds: FloatArray,
    joint_names: tuple[str, ...],
    profile: InitialCollisionProfile,
) -> FloatArray:
    smoothed = qpos.copy(order="C")
    qpos_indices = model.get_qpos_indices(joint_names)
    for joint_name, qpos_index_value in zip(joint_names, qpos_indices, strict=True):
        qpos_index = int(qpos_index_value)
        reference = qpos[:, qpos_index]
        try:
            shaped = shape_trajectory_1d(
                seconds,
                reference,
                tracking_norm=profile.smooth_tracking_norm,
                jerk_weight=profile.smooth_jerk_weight,
            )
            smoothed[:, qpos_index] = shaped.values
        except Exception:
            smoothed[:, qpos_index] = reference

        joint_id = int(mujoco.mj_name2id(model.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if bool(model.model.jnt_limited[joint_id]):
            lower, upper = model.model.jnt_range[joint_id]
            smoothed[:, qpos_index] = np.clip(
                smoothed[:, qpos_index],
                float(lower),
                float(upper),
            )
    return smoothed


def _count_violations(
    model: MujocoModel,
    qpos: FloatArray,
    candidates: CollisionCandidateSet,
    margin: float,
    profile: InitialCollisionProfile,
    backend: BackendSelection,
) -> tuple[int, int]:
    total = 0
    maximum = 0
    for frame_qpos in qpos:
        model.forward(frame_qpos)
        _, indices = _violation_batch(model, candidates, margin, profile, backend)
        count = int(len(indices))
        total += count
        maximum = max(maximum, count)
    return total, maximum


def _diagnostics(
    candidates: CollisionCandidateSet,
    backend: BackendSelection,
    *,
    input_counts: tuple[int, int],
    output_counts: tuple[int, int],
    passes: Sequence[CollisionPassDiagnostics],
) -> InitialCollisionDiagnostics:
    if backend.is_native:
        overall_backend = "native_nanobind"
        distance_backend = "native_mujoco_signed_distance"
        ik_backend = "native_body_position_ik_reference_ordered"
    else:
        overall_backend = "python_reference_ordered"
        distance_backend = "mujoco_signed_distance"
        ik_backend = "python_body_position_ik_reference_ordered"
    return InitialCollisionDiagnostics(
        backend=overall_backend,
        distance_backend=distance_backend,
        ik_backend=ik_backend,
        trajectory_backend="cvxpy_clarabel_with_fallback",
        root_geom_count=candidates.root_geom_count,
        collision_geom_count=candidates.collision_geom_count,
        raw_candidate_pair_count=candidates.raw_pair_count,
        candidate_pair_count=candidates.pair_count,
        arm_joint_names=candidates.arm_joint_names,
        input_violations=input_counts[0],
        input_max_frame_violations=input_counts[1],
        output_violations=output_counts[0],
        output_max_frame_violations=output_counts[1],
        passes=tuple(passes),
    )


def run_initial_collision(
    qpos: ArrayLike,
    seconds: ArrayLike,
    *,
    robot_id: str,
    fps: float,
    progress: InitialCollisionProgress | None = None,
    backend: BackendPreference | BackendSelection = "python",
) -> InitialCollisionResult:
    """Run the faithful Stage 3 arm-only refinement for a selected robot."""

    _validate_reference_environment()
    backend_selection = resolve_backend(backend)
    profile = get_initial_collision_profile(robot_id)
    trajectory, time_values = _validate_inputs(qpos, seconds, fps, profile)
    visible_model = MujocoModel.from_robot(profile.robot_id)
    if int(visible_model.model.nq) != profile.qpos_dim:
        raise ConfigurationError(
            f"Robot {profile.robot_id!r} model nq no longer matches its collision profile."
        )
    candidates = build_collision_candidates(visible_model, profile)
    empty_counts = (0, 0)
    if candidates.pair_count == 0 or not candidates.arm_joint_names:
        diagnostics = _diagnostics(
            candidates,
            backend_selection,
            input_counts=empty_counts,
            output_counts=empty_counts,
            passes=(),
        )
        return InitialCollisionResult(
            robot_id=profile.robot_id,
            fps=float(fps),
            seconds=time_values,
            qpos=trajectory,
            diagnostics=diagnostics,
        )

    input_counts = _count_violations(
        visible_model,
        trajectory,
        candidates,
        profile.initial_margin,
        profile,
        backend_selection,
    )
    if input_counts[0] == 0:
        diagnostics = _diagnostics(
            candidates,
            backend_selection,
            input_counts=input_counts,
            output_counts=input_counts,
            passes=(),
        )
        return InitialCollisionResult(
            robot_id=profile.robot_id,
            fps=float(fps),
            seconds=time_values,
            qpos=trajectory,
            diagnostics=diagnostics,
        )

    ik_model = MujocoModel.from_robot(profile.robot_id)
    solver = BodyPositionIKSolver(
        ik_model,
        max_iterations=profile.solver.max_iterations,
        revolute_step=profile.solver.revolute_step,
        revolute_update_limit=profile.solver.revolute_update_limit,
        damping=profile.solver.damping,
        joint_limit_probe=profile.solver.joint_limit_probe,
        home=visible_model.get_qpos(visible_model.rev_pri_joint_names),
        nullspace_gain=0.5,
        reference_ordered=True,
        backend=backend_selection,
    )

    working = trajectory
    margin = profile.initial_margin
    pass_diagnostics: list[CollisionPassDiagnostics] = []
    for outer_pass in range(profile.outer_passes):
        is_final = outer_pass == profile.outer_passes - 1
        pass_margin = (
            profile.final_pass_margin
            if profile.final_pass_without_smoothing and is_final
            else margin
        )
        refined = _collision_pass(
            visible_model,
            solver,
            working,
            candidates,
            profile,
            pass_margin,
            outer_pass,
            progress,
            backend_selection,
        )
        smooth_this_pass = profile.smooth_each_pass and not (
            profile.final_pass_without_smoothing and is_final
        )
        working = (
            _smooth_arm_joints(
                visible_model,
                refined,
                time_values,
                candidates.arm_joint_names,
                profile,
            )
            if smooth_this_pass
            else refined
        )
        pass_counts = _count_violations(
            visible_model,
            working,
            candidates,
            pass_margin,
            profile,
            backend_selection,
        )
        pass_diagnostics.append(
            CollisionPassDiagnostics(
                margin=pass_margin,
                violations=pass_counts[0],
                max_frame_violations=pass_counts[1],
            )
        )
        if pass_counts[0] == 0:
            break
        margin = min(margin * profile.margin_scale, profile.margin_cap)

    output_counts = _count_violations(
        visible_model,
        working,
        candidates,
        profile.initial_margin,
        profile,
        backend_selection,
    )
    diagnostics = _diagnostics(
        candidates,
        backend_selection,
        input_counts=input_counts,
        output_counts=output_counts,
        passes=pass_diagnostics,
    )
    return InitialCollisionResult(
        robot_id=profile.robot_id,
        fps=float(fps),
        seconds=time_values,
        qpos=working,
        diagnostics=diagnostics,
    )


__all__ = [
    "CollisionPassDiagnostics",
    "InitialCollisionDiagnostics",
    "InitialCollisionProgress",
    "InitialCollisionResult",
    "run_initial_collision",
]
