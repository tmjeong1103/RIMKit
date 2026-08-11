"""Contact-aware foot targets and toe-primary FPA IK.

Robot-specific numerical differences are profile data; no algorithm branch is
keyed by a robot name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import (  # type: ignore[import-untyped]
    binary_dilation,
    gaussian_filter1d,
    median_filter,
)

from core_retarget.kinematics.transforms import transform
from core_retarget.motion.contacts import ContactSchedule
from core_retarget.mujoco.ground import (
    foot_ground_signed_distance,
    pose_dependent_toe_target_z,
)
from core_retarget.mujoco.ik import BodyPositionIKSolver
from core_retarget.mujoco.model import MujocoModel
from core_retarget.native import BackendPreference, BackendSelection, resolve_backend
from core_retarget.optimization.fpa import (
    FpaSolveRecord,
    FpaTrajectoryResult,
    shape_pinned_trajectory,
    shape_remain_trajectory,
    slew_limited_lower_envelope,
    slew_limited_upper_envelope,
    splice_contact_target_velocity,
)
from core_retarget.robots.profiles import get_dmr_profile
from core_retarget.robots.profiles.fpa import FpaProfile, get_fpa_profile
from core_retarget.stages.target_trajectories import TargetTrajectoriesResult

FloatArray = NDArray[np.float64]
FPA_TARGET_SOLVE_LABELS = (
    "stage6.right_toe.remain.x",
    "stage6.right_toe.remain.y",
    "stage6.right_toe.remain.z",
    "stage6.left_toe.remain.x",
    "stage6.left_toe.remain.y",
    "stage6.left_toe.remain.z",
    "stage6.right_foot_yaw.remain",
    "stage6.left_foot_yaw.remain",
)
FPA_IK_SOLVE_LABELS = (
    "stage7.ara_root.pinned.x",
    "stage7.source_base.pinned.x",
    "stage7.ara_root.pinned.y",
    "stage7.source_base.pinned.y",
    "stage7.ara_root.pinned.z",
    "stage7.source_base.pinned.z",
    "stage7.base_correction.pinned.x",
    "stage7.base_correction.pinned.y",
    "stage7.base_correction.pinned.z",
)


class AraResultLike(Protocol):
    robot_id: str
    fps: float
    seconds: FloatArray
    root_ara: FloatArray
    right_ankle_ara: FloatArray
    left_ankle_ara: FloatArray
    right_toe_ara: FloatArray
    left_toe_ara: FloatArray
    toe_floor_target_z: float


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_bool(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def _solve_record(label: str, result: FpaTrajectoryResult) -> FpaSolveRecord:
    return FpaSolveRecord(
        label=label,
        solver=result.solver,
        status=result.status,
        objective=result.objective,
    )


def _validated_solve_records(
    records: tuple[FpaSolveRecord, ...],
    expected_labels: tuple[str, ...],
    *,
    stage: str,
) -> tuple[FpaSolveRecord, ...]:
    immutable = tuple(records)
    if not all(isinstance(record, FpaSolveRecord) for record in immutable):
        raise TypeError(f"{stage} solve_records must contain FpaSolveRecord values")
    labels = tuple(record.label for record in immutable)
    if labels != expected_labels:
        raise ValueError(f"{stage} solve record labels must be {expected_labels}; found {labels}")
    return immutable


@dataclass(frozen=True, slots=True)
class FpaTargetsResult:
    """Immutable Stage 6 target archive plus internal transition gains."""

    robot_id: str
    fps: float
    seconds: FloatArray
    qpos_ara: FloatArray
    right_foot_ara: FloatArray
    left_foot_ara: FloatArray
    right_toe_ara_body: FloatArray
    left_toe_ara_body: FloatArray
    right_toe_reference: FloatArray
    left_toe_reference: FloatArray
    right_toe: FloatArray
    left_toe: FloatArray
    right_foot: FloatArray
    left_foot: FloatArray
    right_ankle: FloatArray
    left_ankle: FloatArray
    right_foot_yaw_ara: FloatArray
    left_foot_yaw_ara: FloatArray
    right_foot_yaw: FloatArray
    left_foot_yaw: FloatArray
    right_floor_weight: FloatArray
    left_floor_weight: FloatArray
    right_sole_clearance: FloatArray
    left_sole_clearance: FloatArray
    right_transition_gain: FloatArray
    left_transition_gain: FloatArray
    solve_records: tuple[FpaSolveRecord, ...]

    def __post_init__(self) -> None:
        robot_profile = get_dmr_profile(self.robot_id)
        fps = float(self.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("Stage 6 fps must be finite and positive")
        object.__setattr__(self, "robot_id", robot_profile.robot_id)
        object.__setattr__(self, "fps", fps)
        seconds = _readonly_float(self.seconds).reshape(-1)
        if len(seconds) == 0 or not np.isfinite(seconds).all():
            raise ValueError("Stage 6 seconds must be a non-empty finite vector")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("Stage 6 seconds must be strictly increasing")
        object.__setattr__(self, "seconds", seconds)
        frame_count = len(seconds)
        vector_fields = (
            "right_foot_ara",
            "left_foot_ara",
            "right_toe_ara_body",
            "left_toe_ara_body",
            "right_toe_reference",
            "left_toe_reference",
            "right_toe",
            "left_toe",
            "right_foot",
            "left_foot",
            "right_ankle",
            "left_ankle",
        )
        for name in vector_fields:
            value = _readonly_float(getattr(self, name))
            if value.shape != (frame_count, 3):
                raise ValueError(f"{name} must have shape ({frame_count}, 3)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        scalar_fields = (
            "right_foot_yaw_ara",
            "left_foot_yaw_ara",
            "right_foot_yaw",
            "left_foot_yaw",
            "right_floor_weight",
            "left_floor_weight",
            "right_sole_clearance",
            "left_sole_clearance",
            "right_transition_gain",
            "left_transition_gain",
        )
        for name in scalar_fields:
            value = _readonly_float(getattr(self, name)).reshape(-1)
            if value.shape != (frame_count,):
                raise ValueError(f"{name} must have shape ({frame_count},)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        qpos = _readonly_float(self.qpos_ara)
        if qpos.shape != (frame_count, robot_profile.qpos_dim):
            raise ValueError(f"qpos_ara must have shape ({frame_count}, {robot_profile.qpos_dim})")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos_ara contains NaN or infinity")
        object.__setattr__(self, "qpos_ara", qpos)
        for name in (
            "right_floor_weight",
            "left_floor_weight",
            "right_transition_gain",
            "left_transition_gain",
        ):
            value = getattr(self, name)
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must lie in [0, 1]")
        object.__setattr__(
            self,
            "solve_records",
            _validated_solve_records(
                self.solve_records,
                FPA_TARGET_SOLVE_LABELS,
                stage="Stage 6",
            ),
        )

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 6 archive arrays."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_ara_array": self.qpos_ara,
            "p_rf_trgt_ara_array": self.right_foot_ara,
            "p_lf_trgt_ara_array": self.left_foot_ara,
            "p_rt_trgt_ara_body_array": self.right_toe_ara_body,
            "p_lt_trgt_ara_body_array": self.left_toe_ara_body,
            "p_rt_trgt_fpa_ref_array": self.right_toe_reference,
            "p_lt_trgt_fpa_ref_array": self.left_toe_reference,
            "p_rt_trgt_fpa_array": self.right_toe,
            "p_lt_trgt_fpa_array": self.left_toe,
            "p_rf_trgt_fpa_array": self.right_foot,
            "p_lf_trgt_fpa_array": self.left_foot,
            "p_ra_trgt_fpa_array": self.right_ankle,
            "p_la_trgt_fpa_array": self.left_ankle,
            "yaw_rf_trgt_ara_array": self.right_foot_yaw_ara,
            "yaw_lf_trgt_ara_array": self.left_foot_yaw_ara,
            "yaw_rf_trgt_fpa_array": self.right_foot_yaw,
            "yaw_lf_trgt_fpa_array": self.left_foot_yaw,
            "r_floor_target_weight_fpa": self.right_floor_weight,
            "l_floor_target_weight_fpa": self.left_floor_weight,
            "r_fpa_toe_sole_clearance_array": self.right_sole_clearance,
            "l_fpa_toe_sole_clearance_array": self.left_sole_clearance,
        }


@dataclass(frozen=True, slots=True)
class FpaIkResult:
    """Immutable Stage 7 output and diagnostics."""

    robot_id: str
    fps: float
    seconds: FloatArray
    qpos: FloatArray
    qpos_before_base_smoothing: FloatArray
    qpos_after_base_smoothing_before_ground: FloatArray
    qpos_ara_before_adaptive_smoothing: FloatArray
    qpos_ara_after_adaptive_smoothing: FloatArray
    ik_error: FloatArray
    ik_error_recovery: FloatArray
    right_contact_weight: FloatArray
    left_contact_weight: FloatArray
    right_control_weight: FloatArray
    left_control_weight: FloatArray
    right_ik_weight: FloatArray
    left_ik_weight: FloatArray
    joint_correction_raw: FloatArray
    joint_correction_smooth: FloatArray
    root_z_correction: FloatArray
    base_smoothing_delta: FloatArray
    ara_smoothing_weight: FloatArray
    ara_smoothing_delta: FloatArray
    ara_adaptive_weight_xyz: FloatArray
    ara_flight_guard_mask: NDArray[np.bool_]
    ground_geometry_correction: FloatArray
    right_ground_distance_pre: FloatArray
    left_ground_distance_pre: FloatArray
    right_ground_distance_post: FloatArray
    left_ground_distance_post: FloatArray
    post_ground_micro_lift: FloatArray
    post_ground_dual_support_lower: FloatArray
    post_ground_dual_recovery_safe_scale: FloatArray
    post_ground_dual_recovery_joint_delta: FloatArray
    solve_records: tuple[FpaSolveRecord, ...]

    def __post_init__(self) -> None:
        robot_profile = get_dmr_profile(self.robot_id)
        fps = float(self.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("Stage 7 fps must be finite and positive")
        object.__setattr__(self, "robot_id", robot_profile.robot_id)
        object.__setattr__(self, "fps", fps)
        seconds = _readonly_float(self.seconds).reshape(-1)
        if len(seconds) == 0 or not np.isfinite(seconds).all():
            raise ValueError("Stage 7 seconds must be a non-empty finite vector")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("Stage 7 seconds must be strictly increasing")
        object.__setattr__(self, "seconds", seconds)
        frame_count = len(seconds)
        qpos_fields = (
            "qpos",
            "qpos_before_base_smoothing",
            "qpos_after_base_smoothing_before_ground",
            "qpos_ara_before_adaptive_smoothing",
            "qpos_ara_after_adaptive_smoothing",
        )
        qpos_width: int | None = None
        for name in qpos_fields:
            value = _readonly_float(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != frame_count:
                raise ValueError(f"{name} must have shape (frames, nq)")
            if qpos_width is None:
                qpos_width = value.shape[1]
            elif value.shape[1] != qpos_width:
                raise ValueError("all Stage 7 qpos arrays must have equal width")
            if value.shape[1] != robot_profile.qpos_dim:
                raise ValueError(f"{name} must have qpos width {robot_profile.qpos_dim}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        scalar_fields = (
            "ik_error",
            "ik_error_recovery",
            "right_contact_weight",
            "left_contact_weight",
            "right_control_weight",
            "left_control_weight",
            "right_ik_weight",
            "left_ik_weight",
            "root_z_correction",
            "ara_smoothing_weight",
            "ground_geometry_correction",
            "right_ground_distance_pre",
            "left_ground_distance_pre",
            "right_ground_distance_post",
            "left_ground_distance_post",
            "post_ground_micro_lift",
            "post_ground_dual_support_lower",
            "post_ground_dual_recovery_safe_scale",
        )
        for name in scalar_fields:
            value = _readonly_float(getattr(self, name)).reshape(-1)
            if value.shape != (frame_count,):
                raise ValueError(f"{name} must have shape ({frame_count},)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        vector_fields = (
            "base_smoothing_delta",
            "ara_smoothing_delta",
            "ara_adaptive_weight_xyz",
        )
        for name in vector_fields:
            value = _readonly_float(getattr(self, name))
            if value.shape != (frame_count, 3):
                raise ValueError(f"{name} must have shape ({frame_count}, 3)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        matrix_fields = (
            "joint_correction_raw",
            "joint_correction_smooth",
            "post_ground_dual_recovery_joint_delta",
        )
        for name in matrix_fields:
            value = _readonly_float(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != frame_count:
                raise ValueError(f"{name} must have shape (frames, joints)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            object.__setattr__(self, name, value)
        if self.joint_correction_raw.shape != self.joint_correction_smooth.shape:
            raise ValueError("raw and smooth joint-correction shapes must match")
        if self.post_ground_dual_recovery_joint_delta.shape != self.joint_correction_raw.shape:
            raise ValueError("dual-recovery and joint-correction shapes must match")
        flight_guard = _readonly_bool(self.ara_flight_guard_mask).reshape(-1)
        if flight_guard.shape != (frame_count,):
            raise ValueError("ara_flight_guard_mask has the wrong frame count")
        object.__setattr__(self, "ara_flight_guard_mask", flight_guard)
        expected_solve_labels = FPA_IK_SOLVE_LABELS if frame_count >= 4 else ()
        object.__setattr__(
            self,
            "solve_records",
            _validated_solve_records(
                self.solve_records,
                expected_solve_labels,
                stage="Stage 7",
            ),
        )

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 7 archive arrays."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_fpa_array": self.qpos,
            "qpos_fpa_before_base_smoothing_array": self.qpos_before_base_smoothing,
            "qpos_fpa_after_base_smoothing_before_ground_array": (
                self.qpos_after_base_smoothing_before_ground
            ),
            "qpos_ara_before_low_speed_smoothing_array": (self.qpos_ara_before_adaptive_smoothing),
            "qpos_ara_after_low_speed_smoothing_array": self.qpos_ara_after_adaptive_smoothing,
            "ik_err_fpa_array": self.ik_error,
            "ik_err_fpa_recovery_array": self.ik_error_recovery,
            "rcontact_weight_fpa": self.right_contact_weight,
            "lcontact_weight_fpa": self.left_contact_weight,
            "rcontact_control_weight_fpa": self.right_control_weight,
            "lcontact_control_weight_fpa": self.left_control_weight,
            "rcontact_ik_weight_fpa": self.right_ik_weight,
            "lcontact_ik_weight_fpa": self.left_ik_weight,
            "fpa_joint_correction_raw_array": self.joint_correction_raw,
            "fpa_joint_correction_smooth_array": self.joint_correction_smooth,
            "fpa_root_z_correction_array": self.root_z_correction,
            "fpa_base_smoothing_delta_array": self.base_smoothing_delta,
            "ara_low_speed_smoothing_weight_array": self.ara_smoothing_weight,
            "ara_low_speed_smoothing_delta_array": self.ara_smoothing_delta,
            "ara_adaptive_smoothing_weight_xyz_array": self.ara_adaptive_weight_xyz,
            "ara_adaptive_flight_guard_mask": self.ara_flight_guard_mask,
            "fpa_ground_geometry_correction_array": self.ground_geometry_correction,
            "r_foot_ground_distance_pre_array": self.right_ground_distance_pre,
            "l_foot_ground_distance_pre_array": self.left_ground_distance_pre,
            "r_foot_ground_distance_post_array": self.right_ground_distance_post,
            "l_foot_ground_distance_post_array": self.left_ground_distance_post,
            "profile_post_ground_micro_lift_array": self.post_ground_micro_lift,
            "profile_post_ground_dual_support_lower_array": self.post_ground_dual_support_lower,
            "profile_post_ground_dual_recovery_safe_scale_array": (
                self.post_ground_dual_recovery_safe_scale
            ),
            "profile_post_ground_dual_recovery_joint_delta_array": (
                self.post_ground_dual_recovery_joint_delta
            ),
        }


@dataclass(frozen=True, slots=True)
class FpaResult:
    targets: FpaTargetsResult
    ik: FpaIkResult

    def __post_init__(self) -> None:
        if self.targets.robot_id != self.ik.robot_id:
            raise ValueError("Stage 6 and Stage 7 robot IDs must match")
        if self.targets.fps != self.ik.fps or not np.array_equal(
            self.targets.seconds,
            self.ik.seconds,
        ):
            raise ValueError("Stage 6 and Stage 7 timelines must match")

    @property
    def qpos(self) -> FloatArray:
        return self.ik.qpos

    @property
    def robot_id(self) -> str:
        return self.ik.robot_id

    @property
    def fps(self) -> float:
        return self.ik.fps

    @property
    def seconds(self) -> FloatArray:
        return self.ik.seconds


def _segments(bounds: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    return [
        np.arange(int(start), int(end), dtype=np.int64)
        for start, end in np.asarray(bounds, dtype=np.int64)
    ]


def _yaw(rotation: FloatArray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _mid_hip(model: MujocoModel, profile_robot_id: str) -> FloatArray:
    profile = get_dmr_profile(profile_robot_id)
    right = model.get_body_transform(profile.joi_bodies["rp"])[:3, 3]
    left = model.get_body_transform(profile.joi_bodies["lp"])[:3, 3]
    return np.asarray(0.5 * (right + left), dtype=np.float64)


def _make_contact_reference(
    reference: FloatArray,
    segments: list[NDArray[np.int64]],
    floor_z: float,
) -> tuple[FloatArray, list[tuple[int, int, FloatArray]]]:
    output = reference.copy()
    anchors: list[tuple[int, int, FloatArray]] = []
    for segment in segments:
        if len(segment) == 0:
            continue
        anchor = np.median(reference[segment], axis=0)
        anchor[2] = float(floor_z)
        output[segment] = anchor.reshape(1, 3)
        anchors.append((int(segment[0]), int(segment[-1]), anchor))
    return output, anchors


def _remain_xyz(
    values: FloatArray,
    segments: list[NDArray[np.int64]],
    seconds: FloatArray,
    profile: FpaProfile,
    *,
    label_prefix: str,
) -> tuple[FloatArray, tuple[FpaSolveRecord, ...]]:
    output = values.copy()
    records: list[FpaSolveRecord] = []
    for axis, axis_name in enumerate(("x", "y", "z")):
        result = shape_remain_trajectory(
            seconds,
            values[:, axis],
            segments,
            jerk_weight=profile.position_smooth_lambda,
            remain_weight=profile.position_remain_weight,
        )
        output[:, axis] = result.values
        records.append(_solve_record(f"{label_prefix}.{axis_name}", result))
    return output, tuple(records)


def _remain_yaw(
    values: FloatArray,
    segments: list[NDArray[np.int64]],
    seconds: FloatArray,
    profile: FpaProfile,
    *,
    label: str,
) -> tuple[FloatArray, FpaSolveRecord]:
    result = shape_remain_trajectory(
        seconds,
        np.unwrap(values),
        segments,
        jerk_weight=profile.yaw_smooth_lambda,
        remain_weight=profile.yaw_remain_weight,
    )
    return result.values, _solve_record(label, result)


def _blend_soft_xy(
    target: FloatArray,
    ara: FloatArray,
    hard_label: NDArray[np.bool_],
    confidence: FloatArray,
) -> FloatArray:
    output = target.copy()
    hard_ticks = np.flatnonzero(hard_label)
    if len(hard_ticks) == 0:
        return output
    soft_ticks = np.flatnonzero((confidence > 0.0) & (confidence < 1.0))
    for tick_value in soft_ticks:
        tick = int(tick_value)
        if hard_label[tick]:
            anchor_tick = tick
        else:
            previous = hard_ticks[hard_ticks < tick]
            anchor_tick = int(previous[-1]) if len(previous) else int(hard_ticks[0])
        weight = float(confidence[tick])
        output[tick, :2] = (1.0 - weight) * ara[tick, :2] + weight * target[anchor_tick, :2]
    return output


def _validate_stage6(
    qpos_stage3: ArrayLike,
    trajectories: TargetTrajectoriesResult,
    ara: AraResultLike,
    contacts: ContactSchedule,
    robot_id: str,
    fps: float,
) -> FloatArray:
    profile = get_dmr_profile(robot_id)
    qpos = np.asarray(qpos_stage3, dtype=np.float64)
    expected = (trajectories.frame_count, profile.qpos_dim)
    if qpos.shape != expected:
        raise ValueError(f"qpos_stage3 must have shape {expected}; found {qpos.shape}")
    if not (
        trajectories.robot_id == ara.robot_id == robot_id
        and contacts.frame_count == trajectories.frame_count
    ):
        raise ValueError("FPA stage inputs do not describe the same robot/motion")
    if not (
        np.array_equal(trajectories.seconds, ara.seconds)
        and np.array_equal(trajectories.seconds, contacts.seconds)
    ):
        raise ValueError("FPA stage timestamps do not match")
    if float(fps) != float(trajectories.fps):
        raise ValueError("FPA fps does not match target trajectories")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos_stage3 contains NaN or infinity")
    return qpos.copy(order="C")


def build_fpa_targets(
    qpos_stage3: ArrayLike,
    trajectories: TargetTrajectoriesResult,
    ara: AraResultLike,
    contacts: ContactSchedule,
    *,
    robot_id: str,
    fps: float,
    source_provider: Literal["kimodo", "gem-x"] = "kimodo",
) -> FpaTargetsResult:
    """Build the exact profile-driven Stage 6 toe-primary targets."""

    qpos_input = _validate_stage6(qpos_stage3, trajectories, ara, contacts, robot_id, fps)
    dmr_profile = get_dmr_profile(robot_id, source_provider=source_provider)
    profile = get_fpa_profile(robot_id, source_provider=source_provider)
    model = MujocoModel.from_robot(robot_id)
    frame_count = len(trajectories.seconds)
    dt = 1.0 / float(fps)
    right_segments = _segments(contacts.right_contact_segments)
    left_segments = _segments(contacts.left_contact_segments)
    right_foot_name = dmr_profile.joi_bodies["rf"]
    left_foot_name = dmr_profile.joi_bodies["lf"]
    right_toe_name = dmr_profile.joi_bodies["rt"]
    left_toe_name = dmr_profile.joi_bodies["lt"]

    qpos_ara = np.empty_like(qpos_input)
    right_foot_ara = np.empty((frame_count, 3), dtype=np.float64)
    left_foot_ara = np.empty((frame_count, 3), dtype=np.float64)
    right_toe_body = np.empty((frame_count, 3), dtype=np.float64)
    left_toe_body = np.empty((frame_count, 3), dtype=np.float64)
    right_yaw_ara = np.empty(frame_count, dtype=np.float64)
    left_yaw_ara = np.empty(frame_count, dtype=np.float64)
    for tick in range(frame_count):
        model.forward(qpos_input[tick])
        pose = model.get_qpos()
        pose[:3] += np.asarray(ara.root_ara[tick]) - _mid_hip(model, robot_id)
        model.forward(pose)
        qpos_ara[tick] = model.get_qpos()
        right_foot_transform = model.get_body_transform(right_foot_name)
        left_foot_transform = model.get_body_transform(left_foot_name)
        right_foot_ara[tick] = right_foot_transform[:3, 3]
        left_foot_ara[tick] = left_foot_transform[:3, 3]
        right_toe_body[tick] = model.get_body_transform(right_toe_name)[:3, 3]
        left_toe_body[tick] = model.get_body_transform(left_toe_name)[:3, 3]
        right_yaw_ara[tick] = _yaw(right_foot_transform[:3, :3])
        left_yaw_ara[tick] = _yaw(left_foot_transform[:3, :3])

    right_target_z_raw, _ = pose_dependent_toe_target_z(
        model,
        qpos_ara,
        toe_body_name=right_toe_name,
        foot_body_name=right_foot_name,
        ground_clearance=profile.sole_ground_clearance,
    )
    left_target_z_raw, _ = pose_dependent_toe_target_z(
        model,
        qpos_ara,
        toe_body_name=left_toe_name,
        foot_body_name=left_foot_name,
        ground_clearance=profile.sole_ground_clearance,
    )
    floor_z = float(ara.toe_floor_target_z)
    samples = np.concatenate(
        (
            right_target_z_raw[contacts.right_confidence >= 0.5] - floor_z,
            left_target_z_raw[contacts.left_confidence >= 0.5] - floor_z,
        )
    )
    clearance = (
        0.025
        if len(samples) == 0
        else float(
            np.clip(
                np.quantile(samples, profile.sole_clearance_quantile),
                profile.sole_clearance_min,
                profile.sole_clearance_max,
            )
        )
    )
    right_target_z = np.full(frame_count, floor_z + clearance, dtype=np.float64)
    left_target_z = np.full(frame_count, floor_z + clearance, dtype=np.float64)

    right_reference, right_anchors = _make_contact_reference(
        np.asarray(ara.right_toe_ara), right_segments, floor_z
    )
    left_reference, left_anchors = _make_contact_reference(
        np.asarray(ara.left_toe_ara), left_segments, floor_z
    )
    right_target, right_position_records = _remain_xyz(
        right_reference,
        right_segments,
        trajectories.seconds,
        profile,
        label_prefix="stage6.right_toe.remain",
    )
    left_target, left_position_records = _remain_xyz(
        left_reference,
        left_segments,
        trajectories.seconds,
        profile,
        label_prefix="stage6.left_toe.remain",
    )
    for start, end, anchor in right_anchors:
        segment = np.arange(start, end + 1)
        right_reference[segment] = anchor
        right_target[segment] = anchor
    for start, end, anchor in left_anchors:
        segment = np.arange(start, end + 1)
        left_reference[segment] = anchor
        left_target[segment] = anchor
    right_reference = _blend_soft_xy(
        right_reference,
        np.asarray(ara.right_toe_ara),
        contacts.right_contact_label,
        contacts.right_confidence,
    )
    left_reference = _blend_soft_xy(
        left_reference,
        np.asarray(ara.left_toe_ara),
        contacts.left_contact_label,
        contacts.left_confidence,
    )
    right_target = _blend_soft_xy(
        right_target,
        np.asarray(ara.right_toe_ara),
        contacts.right_contact_label,
        contacts.right_confidence,
    )
    left_target = _blend_soft_xy(
        left_target,
        np.asarray(ara.left_toe_ara),
        contacts.left_contact_label,
        contacts.left_confidence,
    )
    right_weight = np.clip(contacts.right_confidence, 0.0, 1.0)
    left_weight = np.clip(contacts.left_confidence, 0.0, 1.0)
    right_ara = np.asarray(ara.right_toe_ara)
    left_ara = np.asarray(ara.left_toe_ara)
    for tick in range(frame_count):
        if right_weight[tick] > 0.0:
            value = (1.0 - right_weight[tick]) * right_ara[tick, 2] + right_weight[
                tick
            ] * right_target_z[tick]
            right_reference[tick, 2] = value
            right_target[tick, 2] = value
        if left_weight[tick] > 0.0:
            value = (1.0 - left_weight[tick]) * left_ara[tick, 2] + left_weight[
                tick
            ] * left_target_z[tick]
            left_reference[tick, 2] = value
            left_target[tick, 2] = value

    right_target, right_gain = splice_contact_target_velocity(
        right_target,
        right_segments,
        dt=dt,
        touchdown_blend_time=profile.touchdown_preblend_time,
        touchdown_max_target_delta=profile.touchdown_max_target_delta,
        release_blend_time=profile.toe_velocity_blend_time,
        release_max_target_delta=profile.toe_velocity_max_target_delta,
        max_transition_step=(
            profile.toe_transition_max_step if profile.toe_transition_max_step > 0.0 else None
        ),
    )
    left_target, left_gain = splice_contact_target_velocity(
        left_target,
        left_segments,
        dt=dt,
        touchdown_blend_time=profile.touchdown_preblend_time,
        touchdown_max_target_delta=profile.touchdown_max_target_delta,
        release_blend_time=profile.toe_velocity_blend_time,
        release_max_target_delta=profile.toe_velocity_max_target_delta,
        max_transition_step=(
            profile.toe_transition_max_step if profile.toe_transition_max_step > 0.0 else None
        ),
    )
    right_yaw, right_yaw_record = _remain_yaw(
        right_yaw_ara,
        right_segments,
        trajectories.seconds,
        profile,
        label="stage6.right_foot_yaw.remain",
    )
    left_yaw, left_yaw_record = _remain_yaw(
        left_yaw_ara,
        left_segments,
        trajectories.seconds,
        profile,
        label="stage6.left_foot_yaw.remain",
    )

    return FpaTargetsResult(
        robot_id=robot_id,
        fps=float(fps),
        seconds=trajectories.seconds,
        qpos_ara=qpos_ara,
        right_foot_ara=right_foot_ara,
        left_foot_ara=left_foot_ara,
        right_toe_ara_body=right_toe_body,
        left_toe_ara_body=left_toe_body,
        right_toe_reference=right_reference,
        left_toe_reference=left_reference,
        right_toe=right_target,
        left_toe=left_target,
        right_foot=right_foot_ara,
        left_foot=left_foot_ara,
        right_ankle=ara.right_ankle_ara,
        left_ankle=ara.left_ankle_ara,
        right_foot_yaw_ara=right_yaw_ara,
        left_foot_yaw_ara=left_yaw_ara,
        right_foot_yaw=right_yaw,
        left_foot_yaw=left_yaw,
        right_floor_weight=right_weight,
        left_floor_weight=left_weight,
        right_sole_clearance=right_target_z - floor_z,
        left_sole_clearance=left_target_z - floor_z,
        right_transition_gain=right_gain,
        left_transition_gain=left_gain,
        solve_records=(
            *right_position_records,
            *left_position_records,
            right_yaw_record,
            left_yaw_record,
        ),
    )


@dataclass(frozen=True, slots=True)
class _FpaBodies:
    right_toe: str
    left_toe: str
    right_foot: str
    left_foot: str
    right_knee: str
    left_knee: str


@dataclass(frozen=True, slots=True)
class _FpaJointGroups:
    all: tuple[str, ...]
    recovery: tuple[str, ...]
    left_recovery: tuple[str, ...]
    right_recovery: tuple[str, ...]
    recovery_indices: NDArray[np.int32]


@dataclass(slots=True)
class _PrimaryIkState:
    qpos: FloatArray
    error: FloatArray
    right_contact_weight: FloatArray
    left_contact_weight: FloatArray
    right_control_weight: FloatArray
    left_control_weight: FloatArray


@dataclass(slots=True)
class _BaseRecoveryState:
    qpos: FloatArray
    qpos_before_smoothing: FloatArray
    qpos_ara_before_smoothing: FloatArray
    qpos_ara_after_smoothing: FloatArray
    recovery_error: FloatArray
    joint_correction_raw: FloatArray
    joint_correction_smooth: FloatArray
    root_z_correction: FloatArray
    base_smoothing_delta: FloatArray
    ara_smoothing_weight: FloatArray
    ara_smoothing_delta: FloatArray
    ara_adaptive_weight_xyz: FloatArray
    ara_flight_guard_mask: NDArray[np.bool_]
    solve_records: tuple[FpaSolveRecord, ...]


@dataclass(slots=True)
class _GroundState:
    qpos: FloatArray
    qpos_before_ground: FloatArray
    geometry_correction: FloatArray
    right_distance_pre: FloatArray
    left_distance_pre: FloatArray
    right_distance_post: FloatArray
    left_distance_post: FloatArray
    micro_lift: FloatArray
    dual_support_lower: FloatArray
    dual_recovery_safe_scale: FloatArray
    dual_recovery_joint_delta: FloatArray


def _has_token(name: str, tokens: tuple[str, ...]) -> bool:
    lower_name = name.lower()
    return any(token.lower() in lower_name for token in tokens)


def _has_side_token(name: str, tokens: tuple[str, ...]) -> bool:
    """Match side labels without confusing ``_R`` with ``_ROLL``."""

    lower_name = name.lower()
    return any(
        lower_name.endswith(token.lower())
        if token.startswith("_")
        else token.lower() in lower_name
        for token in tokens
    )


def _fpa_bodies(robot_id: str) -> _FpaBodies:
    mapping = get_dmr_profile(robot_id).joi_bodies
    return _FpaBodies(
        right_toe=mapping["rt"],
        left_toe=mapping["lt"],
        right_foot=mapping["rf"],
        left_foot=mapping["lf"],
        right_knee=mapping["rk"],
        left_knee=mapping["lk"],
    )


def _fpa_joint_groups(
    model: MujocoModel,
    profile: FpaProfile,
) -> _FpaJointGroups:
    all_names = tuple(
        str(name)
        for name in model.rev_pri_joint_names
        if not _has_token(str(name), profile.excluded_joint_tokens)
    )
    recovery = tuple(name for name in all_names if _has_token(name, profile.recovery_joint_tokens))
    left = tuple(name for name in recovery if _has_side_token(name, profile.left_joint_tokens))
    right = tuple(name for name in recovery if _has_side_token(name, profile.right_joint_tokens))
    if not recovery or not left or not right:
        raise ValueError("FPA could not resolve the profile's leg recovery joints")
    return _FpaJointGroups(
        all=all_names,
        recovery=recovery,
        left_recovery=left,
        right_recovery=right,
        recovery_indices=model.get_qpos_indices(recovery),
    )


def _make_fpa_solver(
    model: MujocoModel,
    backend: BackendPreference | BackendSelection = "python",
) -> BodyPositionIKSolver:
    return BodyPositionIKSolver(
        model,
        max_iterations=80,
        revolute_step=0.5,
        prismatic_step=0.0,
        revolute_update_limit=np.deg2rad(5.0),
        prismatic_update_limit=0.0,
        damping=1e-4,
        joint_limit_probe=np.deg2rad(3.0),
        prismatic_limit_probe=0.0,
        nullspace_gain=0.0,
        home=model.get_qpos(model.rev_pri_joint_names),
        reference_ordered=True,
        backend=backend,
    )


def _rpy(rotation: FloatArray) -> FloatArray:
    return np.asarray(
        (
            np.atan2(rotation[2, 1], rotation[2, 2]),
            np.atan2(
                -rotation[2, 0],
                np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2),
            ),
            np.atan2(rotation[1, 0], rotation[0, 0]),
        ),
        dtype=np.float64,
    )


def _rotation_from_rpy(values: ArrayLike) -> FloatArray:
    roll, pitch, yaw = np.asarray(values, dtype=np.float64).reshape(3)
    c_roll, s_roll = np.cos(roll), np.sin(roll)
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (
                c_yaw * c_pitch,
                -s_yaw * c_roll + c_yaw * s_pitch * s_roll,
                s_yaw * s_roll + c_yaw * s_pitch * c_roll,
            ),
            (
                s_yaw * c_pitch,
                c_yaw * c_roll + s_yaw * s_pitch * s_roll,
                -c_yaw * s_roll + s_yaw * s_pitch * c_roll,
            ),
            (-s_pitch, c_pitch * s_roll, c_pitch * c_roll),
        ),
        dtype=np.float64,
    )


def _add_primary_targets(
    solver: BodyPositionIKSolver,
    model: MujocoModel,
    bodies: _FpaBodies,
    targets: FpaTargetsResult,
    tick: int,
    right_weight: float,
    left_weight: float,
    profile: FpaProfile,
    right_orientation_body: str | None,
    left_orientation_body: str | None,
    right_ankle_rotations: FloatArray | None,
    left_ankle_rotations: FloatArray | None,
) -> None:
    right_toe_transform = model.get_body_transform(bodies.right_toe)
    left_toe_transform = model.get_body_transform(bodies.left_toe)
    right_foot_transform = model.get_body_transform(bodies.right_foot)
    left_foot_transform = model.get_body_transform(bodies.left_foot)
    right_knee = model.get_body_transform(bodies.right_knee)[:3, 3]
    left_knee = model.get_body_transform(bodies.left_knee)[:3, 3]
    right_rpy = _rpy(right_foot_transform[:3, :3])
    left_rpy = _rpy(left_foot_transform[:3, :3])
    right_target_transform = transform(
        targets.right_foot[tick],
        _rotation_from_rpy((right_rpy[0], right_rpy[1], targets.right_foot_yaw[tick])),
    )
    left_target_transform = transform(
        targets.left_foot[tick],
        _rotation_from_rpy((left_rpy[0], left_rpy[1], targets.left_foot_yaw[tick])),
    )

    solver.reset_targets(sync_from=model)
    solver.add_target(
        bodies.right_toe,
        right_toe_transform[:3, 3],
        targets.right_toe[tick],
        1.0 + 7.0 * right_weight,
    )
    solver.add_target(
        bodies.left_toe,
        left_toe_transform[:3, 3],
        targets.left_toe[tick],
        1.0 + 7.0 * left_weight,
    )
    solver.add_target(
        bodies.right_foot,
        right_foot_transform[:3, 3],
        targets.right_foot[tick],
        0.15,
    )
    solver.add_target(
        bodies.left_foot,
        left_foot_transform[:3, 3],
        targets.left_foot[tick],
        0.15,
    )
    if (
        right_orientation_body is not None
        and left_orientation_body is not None
        and right_ankle_rotations is not None
        and left_ankle_rotations is not None
    ):
        right_orientation = model.get_body_transform(right_orientation_body)
        left_orientation = model.get_body_transform(left_orientation_body)
        right_profile_rpy = _rpy(right_ankle_rotations[tick])
        left_profile_rpy = _rpy(left_ankle_rotations[tick])
        solver.add_transform_target(
            right_orientation_body,
            right_orientation,
            transform(
                right_orientation[:3, 3],
                _rotation_from_rpy(
                    (right_profile_rpy[0], right_profile_rpy[1], targets.right_foot_yaw[tick])
                ),
            ),
            weight=0.05 + right_weight * profile.contact_ankle_orientation_weight,
            axis_length=0.25,
        )
        solver.add_transform_target(
            left_orientation_body,
            left_orientation,
            transform(
                left_orientation[:3, 3],
                _rotation_from_rpy(
                    (left_profile_rpy[0], left_profile_rpy[1], targets.left_foot_yaw[tick])
                ),
            ),
            weight=0.05 + left_weight * profile.contact_ankle_orientation_weight,
            axis_length=0.25,
        )
    else:
        solver.add_transform_target(
            bodies.right_foot,
            right_foot_transform,
            right_target_transform,
            weight=0.05,
            axis_length=0.25,
        )
        solver.add_transform_target(
            bodies.left_foot,
            left_foot_transform,
            left_target_transform,
            weight=0.05,
            axis_length=0.25,
        )
    solver.add_target(bodies.right_knee, right_knee, right_knee, 0.35)
    solver.add_target(bodies.left_knee, left_knee, left_knee, 0.35)


def _run_primary_ik(
    model: MujocoModel,
    solver: BodyPositionIKSolver,
    bodies: _FpaBodies,
    groups: _FpaJointGroups,
    targets: FpaTargetsResult,
    contacts: ContactSchedule,
    profile: FpaProfile,
    right_orientation_body: str | None = None,
    left_orientation_body: str | None = None,
    right_ankle_rotations: FloatArray | None = None,
    left_ankle_rotations: FloatArray | None = None,
) -> _PrimaryIkState:
    frame_count = len(targets.seconds)
    qpos = np.zeros_like(targets.qpos_ara)
    error = np.zeros(frame_count, dtype=np.float64)
    right_contact = np.clip(np.asarray(contacts.right_confidence, dtype=np.float64), 0.0, 1.0)
    left_contact = np.clip(np.asarray(contacts.left_confidence, dtype=np.float64), 0.0, 1.0)
    right_control = np.clip(np.maximum(right_contact, targets.right_transition_gain), 0.0, 1.0)
    left_control = np.clip(np.maximum(left_contact, targets.left_transition_gain), 0.0, 1.0)
    for tick in range(frame_count):
        model.forward(targets.qpos_ara[tick])
        _add_primary_targets(
            solver,
            model,
            bodies,
            targets,
            tick,
            float(right_control[tick]),
            float(left_control[tick]),
            profile,
            right_orientation_body,
            left_orientation_body,
            right_ankle_rotations,
            left_ankle_rotations,
        )
        result = solver.solve(
            joints=groups.all,
            joint_limits=True,
            nullspace=False,
            base_control=False,
        )
        model.forward(result.qpos)
        xyz_errors: list[FloatArray] = []
        xy_weights: list[float] = []
        if right_control[tick] > 0.0:
            xyz_errors.append(
                model.get_body_transform(bodies.right_toe)[:3, 3] - targets.right_toe[tick]
            )
            xy_weights.append(float(right_control[tick]))
        if left_control[tick] > 0.0:
            xyz_errors.append(
                model.get_body_transform(bodies.left_toe)[:3, 3] - targets.left_toe[tick]
            )
            xy_weights.append(float(left_control[tick]))
        if xyz_errors:
            correction_gain = max(float(right_control[tick]), float(left_control[tick]))
            correction = np.average(np.asarray(xyz_errors), axis=0, weights=np.asarray(xy_weights))
            corrected = model.get_qpos()
            corrected[:2] -= correction_gain * correction[:2]
            model.forward(corrected)
        qpos[tick] = model.get_qpos()
        error[tick] = result.error
    return _PrimaryIkState(
        qpos=qpos,
        error=error,
        right_contact_weight=right_contact,
        left_contact_weight=left_contact,
        right_control_weight=right_control,
        left_control_weight=left_control,
    )


def _smooth_joint_correction(
    qpos: FloatArray,
    qpos_ara: FloatArray,
    groups: _FpaJointGroups,
    profile: FpaProfile,
    right_contact: FloatArray,
    left_contact: FloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    indices = groups.recovery_indices
    raw = qpos[:, indices] - qpos_ara[:, indices]
    smooth = raw.copy()
    enabled = (
        profile.joint_correction_median_window > 1 or profile.joint_correction_smooth_time > 0.0
    )
    if not enabled:
        return raw, smooth
    window = max(1, int(profile.joint_correction_median_window))
    if window % 2 == 0:
        window += 1
    candidate = median_filter(raw, size=(window, 1), mode="nearest")
    if profile.joint_correction_smooth_time > 0.0:
        sigma = max(profile.joint_correction_smooth_time / max(dt, 1e-12), 1e-6)
        candidate = gaussian_filter1d(candidate, sigma=sigma, axis=0, mode="nearest")
    adjustment_full = candidate - raw
    adjustment_scale = np.ones_like(raw)
    if profile.joint_correction_max_delta > 0.0:
        adjustment_scale = np.minimum(
            1.0,
            profile.joint_correction_max_delta / np.maximum(np.abs(adjustment_full), 1e-12),
        )
    adjustment = adjustment_full * adjustment_scale
    if profile.swing_outlier_max_adjustment > 0.0:
        contact_confidence = np.ones_like(raw)
        left_set = set(groups.left_recovery)
        right_set = set(groups.right_recovery)
        for column, joint_name in enumerate(groups.recovery):
            if joint_name in left_set:
                contact_confidence[:, column] = left_contact
            elif joint_name in right_set:
                contact_confidence[:, column] = right_contact
        outlier = (contact_confidence < profile.swing_outlier_contact_threshold) & (
            np.abs(adjustment_full) > profile.swing_outlier_threshold
        )
        outlier_adjustment = np.clip(
            adjustment_full,
            -profile.swing_outlier_max_adjustment,
            profile.swing_outlier_max_adjustment,
        )
        adjustment = np.where(outlier, outlier_adjustment, adjustment)
    smooth = raw + adjustment
    qpos[:, indices] = qpos_ara[:, indices] + smooth
    return raw, smooth


def _apply_root_z_correction(
    model: MujocoModel,
    bodies: _FpaBodies,
    qpos: FloatArray,
    targets: FpaTargetsResult,
    right_control: FloatArray,
    left_control: FloatArray,
) -> FloatArray:
    samples = np.full(len(qpos), np.nan, dtype=np.float64)
    for tick in range(len(qpos)):
        model.forward(qpos[tick])
        errors: list[float] = []
        weights: list[float] = []
        if right_control[tick] > 0.0:
            errors.append(
                float(model.get_body_transform(bodies.right_toe)[2, 3] - targets.right_toe[tick, 2])
            )
            weights.append(float(right_control[tick]))
        if left_control[tick] > 0.0:
            errors.append(
                float(model.get_body_transform(bodies.left_toe)[2, 3] - targets.left_toe[tick, 2])
            )
            weights.append(float(left_control[tick]))
        if errors:
            confidence_gain = min(1.0, float(np.sum(weights)))
            samples[tick] = confidence_gain * float(
                np.average(np.asarray(errors), weights=np.asarray(weights))
            )
    ticks = np.flatnonzero(np.isfinite(samples))
    correction: FloatArray = np.zeros(len(qpos), dtype=np.float64)
    if len(ticks):
        correction = np.interp(
            np.arange(len(qpos), dtype=np.float64),
            ticks.astype(np.float64),
            samples[ticks],
        )
        qpos[:, 2] -= correction
    return correction


def _extract_recovery_references(
    model: MujocoModel,
    bodies: _FpaBodies,
    qpos: FloatArray,
) -> dict[str, FloatArray]:
    references: dict[str, FloatArray] = {
        name: np.zeros((len(qpos), 3), dtype=np.float64)
        for name in (
            "right_toe",
            "left_toe",
            "right_foot",
            "left_foot",
            "right_knee",
            "left_knee",
        )
    }
    body_names = {
        "right_toe": bodies.right_toe,
        "left_toe": bodies.left_toe,
        "right_foot": bodies.right_foot,
        "left_foot": bodies.left_foot,
        "right_knee": bodies.right_knee,
        "left_knee": bodies.left_knee,
    }
    for tick in range(len(qpos)):
        model.forward(qpos[tick])
        for key, body_name in body_names.items():
            references[key][tick] = model.get_body_transform(body_name)[:3, 3]
    return references


def _adaptive_ara_smoothing(
    qpos_ara: FloatArray,
    contacts: ContactSchedule,
    right_contact: FloatArray,
    left_contact: FloatArray,
    seconds: FloatArray,
    dt: float,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.bool_],
    tuple[FpaSolveRecord, ...],
]:
    before = qpos_ara.copy()
    after = qpos_ara.copy()
    frame_count = len(qpos_ara)
    smoothing_weight = np.zeros(frame_count, dtype=np.float64)
    smoothing_delta = np.zeros((frame_count, 3), dtype=np.float64)
    adaptive_weight = np.zeros((frame_count, 3), dtype=np.float64)
    flight_guard = np.zeros(frame_count, dtype=np.bool_)
    if frame_count < 4:
        return (
            after,
            smoothing_weight,
            smoothing_delta,
            adaptive_weight,
            flight_guard,
            (),
        )

    support_weight = np.maximum(right_contact, left_contact)
    flight_mask = (
        np.zeros(frame_count, dtype=np.bool_)
        if contacts.flight_label is None
        else np.asarray(contacts.flight_label, dtype=np.bool_)
    )
    flight_guard = binary_dilation(flight_mask, iterations=3)
    mask_seed = support_weight * (~flight_guard).astype(np.float64)
    feather_sigma = max(0.06 / max(dt, 1e-12), 1e-6)
    support_feather = np.clip(
        gaussian_filter1d(mask_seed, sigma=feather_sigma, mode="nearest"),
        0.0,
        1.0,
    )
    support_feather[flight_guard] = 0.0
    position_raw = before[:, :3]
    source_reference = np.asarray(contacts.base_positions_smoothed, dtype=np.float64).copy()
    rms_sigma = max(0.10 / max(dt, 1e-12), 1e-6)
    maximum_delta = np.asarray((0.003, 0.003, 0.002), dtype=np.float64)
    records: list[FpaSolveRecord] = []
    for axis, axis_name in enumerate(("x", "y", "z")):
        axis_reference = position_raw[:, axis]
        trend_result = shape_pinned_trajectory(seconds, axis_reference, jerk_weight=3e-8)
        trend = trend_result.values
        records.append(_solve_record(f"stage7.ara_root.pinned.{axis_name}", trend_result))
        source_axis = source_reference[:, axis]
        source_trend_result = shape_pinned_trajectory(seconds, source_axis, jerk_weight=3e-8)
        source_trend = source_trend_result.values
        records.append(
            _solve_record(f"stage7.source_base.pinned.{axis_name}", source_trend_result)
        )
        residual = axis_reference - trend
        source_residual = source_axis - source_trend
        residual_rms = np.sqrt(
            np.maximum(
                gaussian_filter1d(residual * residual, sigma=rms_sigma, mode="nearest"),
                0.0,
            )
        )
        source_residual_rms = np.sqrt(
            np.maximum(
                gaussian_filter1d(
                    source_residual * source_residual,
                    sigma=rms_sigma,
                    mode="nearest",
                ),
                0.0,
            )
        )
        excess_weight = np.clip(
            (residual_rms - 1.5 * source_residual_rms - 0.0001) / np.maximum(residual_rms, 1e-12),
            0.0,
            1.0,
        )
        axis_weight = support_feather * excess_weight
        axis_weight[flight_guard] = 0.0
        masked_delta = axis_weight * (trend - axis_reference)
        maximum = float(np.max(np.abs(masked_delta)))
        scale = min(1.0, maximum_delta[axis] / max(maximum, 1e-12))
        smoothing_delta[:, axis] = scale * masked_delta
        adaptive_weight[:, axis] = axis_weight
    smoothing_weight = np.max(adaptive_weight, axis=1)
    after[:, :3] = position_raw + smoothing_delta
    return (
        after,
        smoothing_weight,
        smoothing_delta,
        adaptive_weight,
        flight_guard,
        tuple(records),
    )


def _smooth_base_correction(
    qpos_before: FloatArray,
    qpos_ara: FloatArray,
    qpos_ara_after: FloatArray,
    seconds: FloatArray,
) -> tuple[FloatArray, FloatArray, tuple[FpaSolveRecord, ...]]:
    raw = qpos_before[:, :3] - qpos_ara[:, :3]
    smooth = raw.copy()
    records: list[FpaSolveRecord] = []
    if len(qpos_before) >= 4:
        caps = np.asarray((0.008, 0.008, 0.005), dtype=np.float64)
        for axis, axis_name in enumerate(("x", "y", "z")):
            reference = raw[:, axis]
            result = shape_pinned_trajectory(seconds, reference, jerk_weight=1e-8)
            candidate = result.values
            records.append(_solve_record(f"stage7.base_correction.pinned.{axis_name}", result))
            delta = candidate - reference
            maximum = float(np.max(np.abs(delta)))
            scale = min(1.0, caps[axis] / max(maximum, 1e-12))
            smooth[:, axis] = reference + scale * delta
    smoothing_delta = smooth - raw
    root_target = qpos_ara_after[:, :3] + smooth
    return root_target, smoothing_delta, tuple(records)


def _add_recovery_targets(
    solver: BodyPositionIKSolver,
    model: MujocoModel,
    bodies: _FpaBodies,
    references: dict[str, FloatArray],
    tick: int,
    right_weight: float,
    left_weight: float,
) -> None:
    right_toe_transform = model.get_body_transform(bodies.right_toe)
    left_toe_transform = model.get_body_transform(bodies.left_toe)
    right_foot_transform = model.get_body_transform(bodies.right_foot)
    left_foot_transform = model.get_body_transform(bodies.left_foot)
    solver.reset_targets(sync_from=model)
    solver.add_target(
        bodies.right_toe,
        right_toe_transform[:3, 3],
        references["right_toe"][tick],
        1.0 + 7.0 * right_weight,
    )
    solver.add_target(
        bodies.left_toe,
        left_toe_transform[:3, 3],
        references["left_toe"][tick],
        1.0 + 7.0 * left_weight,
    )
    solver.add_target(
        bodies.right_foot,
        right_foot_transform[:3, 3],
        references["right_foot"][tick],
        0.15,
    )
    solver.add_target(
        bodies.left_foot,
        left_foot_transform[:3, 3],
        references["left_foot"][tick],
        0.15,
    )
    solver.add_transform_target(
        bodies.right_foot,
        right_foot_transform,
        transform(references["right_foot"][tick], right_foot_transform[:3, :3]),
        weight=0.05,
        axis_length=0.25,
    )
    solver.add_transform_target(
        bodies.left_foot,
        left_foot_transform,
        transform(references["left_foot"][tick], left_foot_transform[:3, :3]),
        weight=0.05,
        axis_length=0.25,
    )
    right_knee = model.get_body_transform(bodies.right_knee)[:3, 3]
    left_knee = model.get_body_transform(bodies.left_knee)[:3, 3]
    solver.add_target(
        bodies.right_knee,
        right_knee,
        references["right_knee"][tick],
        0.35,
    )
    solver.add_target(
        bodies.left_knee,
        left_knee,
        references["left_knee"][tick],
        0.35,
    )


def _run_base_recovery(
    model: MujocoModel,
    solver: BodyPositionIKSolver,
    bodies: _FpaBodies,
    groups: _FpaJointGroups,
    profile: FpaProfile,
    targets: FpaTargetsResult,
    contacts: ContactSchedule,
    primary: _PrimaryIkState,
    dt: float,
) -> _BaseRecoveryState:
    qpos = primary.qpos.copy()
    correction_raw, correction_smooth = _smooth_joint_correction(
        qpos,
        targets.qpos_ara,
        groups,
        profile,
        primary.right_contact_weight,
        primary.left_contact_weight,
        dt,
    )
    root_z = _apply_root_z_correction(
        model,
        bodies,
        qpos,
        targets,
        primary.right_control_weight,
        primary.left_control_weight,
    )
    before_smoothing = qpos.copy()
    references = _extract_recovery_references(model, bodies, before_smoothing)
    qpos_ara_before = targets.qpos_ara.copy()
    (
        qpos_ara_after,
        ara_weight,
        ara_delta,
        adaptive_weight,
        flight_guard,
        adaptive_solve_records,
    ) = _adaptive_ara_smoothing(
        targets.qpos_ara,
        contacts,
        primary.right_contact_weight,
        primary.left_contact_weight,
        targets.seconds,
        dt,
    )
    root_target, base_delta, base_solve_records = _smooth_base_correction(
        before_smoothing,
        targets.qpos_ara,
        qpos_ara_after,
        targets.seconds,
    )
    qpos[:, :3] = root_target
    recovery_error = np.full(len(qpos), np.nan, dtype=np.float64)
    if profile.post_ground_recovery_passes == 0:
        for tick in range(len(qpos)):
            model.forward(qpos[tick])
            _add_recovery_targets(
                solver,
                model,
                bodies,
                references,
                tick,
                float(primary.right_contact_weight[tick]),
                float(primary.left_contact_weight[tick]),
            )
            result = solver.solve(
                joints=groups.recovery,
                joint_limits=True,
                nullspace=False,
                base_control=False,
            )
            recovered = np.asarray(result.qpos, dtype=np.float64).copy()
            reference_joints = before_smoothing[tick, groups.recovery_indices]
            joint_delta = recovered[groups.recovery_indices] - reference_joints
            maximum = float(np.max(np.abs(joint_delta)))
            scale = min(1.0, np.deg2rad(3.0) / max(maximum, 1e-12))
            recovered[groups.recovery_indices] = reference_joints + scale * joint_delta
            model.forward(recovered)
            qpos[tick] = model.get_qpos()
            recovery_error[tick] = result.error
    root_drift = float(np.max(np.linalg.norm(qpos[:, :3] - root_target, axis=1)))
    if root_drift > 1e-8:
        raise RuntimeError(
            f"Fixed-base FPA recovery changed root translation by {root_drift:.3g} m"
        )
    return _BaseRecoveryState(
        qpos=qpos,
        qpos_before_smoothing=before_smoothing,
        qpos_ara_before_smoothing=qpos_ara_before,
        qpos_ara_after_smoothing=qpos_ara_after,
        recovery_error=recovery_error,
        joint_correction_raw=correction_raw,
        joint_correction_smooth=correction_smooth,
        root_z_correction=root_z,
        base_smoothing_delta=base_delta,
        ara_smoothing_weight=ara_weight,
        ara_smoothing_delta=ara_delta,
        ara_adaptive_weight_xyz=adaptive_weight,
        ara_flight_guard_mask=flight_guard,
        solve_records=(*adaptive_solve_records, *base_solve_records),
    )


def _ground_distances(
    model: MujocoModel,
    bodies: _FpaBodies,
    qpos: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    return (
        foot_ground_signed_distance(model, qpos, foot_body_name=bodies.right_foot),
        foot_ground_signed_distance(model, qpos, foot_body_name=bodies.left_foot),
    )


def _post_ground_evaluation(
    model: MujocoModel,
    bodies: _FpaBodies,
    qpos: FloatArray,
    right_weight: float,
    left_weight: float,
) -> tuple[float, bool, FloatArray]:
    right_distance, left_distance = _ground_distances(
        model, bodies, np.asarray(qpos, dtype=np.float64)[None, :]
    )
    right_value = float(right_distance[0])
    left_value = float(left_distance[0])
    components = np.asarray(
        (
            right_weight * max(right_value - 0.001, 0.0) ** 2,
            left_weight * max(left_value - 0.001, 0.0) ** 2,
        ),
        dtype=np.float64,
    )
    safe = bool(
        (right_weight < 0.5 or right_value >= 0.001 - 1e-6)
        and (left_weight < 0.5 or left_value >= 0.001 - 1e-6)
    )
    return float(np.sum(components)), safe, components


def _post_ground_improves(
    model: MujocoModel,
    bodies: _FpaBodies,
    qpos: FloatArray,
    right_weight: float,
    left_weight: float,
    base_error: float,
    base_components: FloatArray,
) -> bool:
    candidate_error, safe, candidate_components = _post_ground_evaluation(
        model, bodies, qpos, right_weight, left_weight
    )
    active = np.asarray((right_weight, left_weight), dtype=np.float64) > 1e-6
    components_ok = bool(np.all(candidate_components[active] <= base_components[active] + 1e-12))
    return bool(safe and components_ok and candidate_error < base_error - 1e-12)


def _run_dual_recovery(
    model: MujocoModel,
    solver: BodyPositionIKSolver,
    bodies: _FpaBodies,
    groups: _FpaJointGroups,
    profile: FpaProfile,
    qpos: FloatArray,
    right_contact: FloatArray,
    left_contact: FloatArray,
    dual_support_lower: FloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    safe_scale = np.ones(len(qpos), dtype=np.float64)
    accumulated_delta = np.zeros((len(qpos), len(groups.recovery_indices)), dtype=np.float64)
    for _ in range(profile.post_ground_dual_recovery_passes):
        right_ground, left_ground = _ground_distances(model, bodies, qpos)
        raw_delta = np.zeros_like(accumulated_delta)
        for tick in range(len(qpos)):
            dual_weight = float(min(right_contact[tick], left_contact[tick]))
            if dual_weight <= 1e-6:
                continue
            if dual_support_lower[tick] < profile.post_ground_dual_recovery_min_root_lower:
                continue
            deadband = profile.post_ground_dual_recovery_height_deadband
            use_right = bool(
                right_ground[tick] > 0.001 + deadband
                and right_ground[tick] > left_ground[tick] + deadband
            )
            use_left = bool(
                left_ground[tick] > 0.001 + deadband
                and left_ground[tick] > right_ground[tick] + deadband
            )
            if not (use_right or use_left):
                continue
            selected_gap = float(right_ground[tick]) if use_right else float(left_ground[tick])
            selected_dz = dual_weight * float(np.clip(0.001 - selected_gap, -0.03, 0.0))
            model.forward(qpos[tick])
            q_before = model.get_qpos()
            toe_name = bodies.right_toe if use_right else bodies.left_toe
            foot_name = bodies.right_foot if use_right else bodies.left_foot
            toe_position = model.get_body_transform(toe_name)[:3, 3]
            foot_position = model.get_body_transform(foot_name)[:3, 3]
            toe_target = toe_position.copy()
            foot_target = foot_position.copy()
            toe_target[2] += selected_dz
            foot_target[2] += selected_dz
            solver.reset_targets(sync_from=model)
            solver.add_target(
                toe_name,
                toe_position,
                toe_target,
                weight=6.0 * dual_weight,
            )
            solver.add_target(
                foot_name,
                foot_position,
                foot_target,
                weight=0.5 * dual_weight,
            )
            result = solver.solve(
                joints=groups.recovery,
                joint_limits=True,
                nullspace=False,
                base_control=False,
            )
            joint_delta = result.qpos[groups.recovery_indices] - q_before[groups.recovery_indices]
            maximum = float(np.max(np.abs(joint_delta)))
            scale = min(
                1.0,
                profile.post_ground_dual_recovery_joint_delta / max(maximum, 1e-12),
            )
            raw_delta[tick] = scale * joint_delta
        if profile.post_ground_recovery_smooth_time > 0.0 and len(qpos) > 1:
            sigma = max(
                profile.post_ground_recovery_smooth_time / max(dt, 1e-12),
                1e-6,
            )
            raw_delta = gaussian_filter1d(raw_delta, sigma=sigma, axis=0, mode="nearest")
        recovered = qpos.copy()
        for tick in range(len(qpos)):
            joint_delta = raw_delta[tick]
            if float(np.max(np.abs(joint_delta))) <= 1e-12:
                continue
            q_base = qpos[tick].copy()
            right_weight = float(right_contact[tick])
            left_weight = float(left_contact[tick])
            base_error, _, base_components = _post_ground_evaluation(
                model,
                bodies,
                q_base,
                right_weight,
                left_weight,
            )
            accepted_scale = 0.0
            for line_step in range(profile.post_ground_dual_recovery_safety_iterations + 1):
                candidate_scale = 0.5**line_step
                candidate = q_base.copy()
                candidate[groups.recovery_indices] += candidate_scale * joint_delta
                if _post_ground_improves(
                    model,
                    bodies,
                    candidate,
                    right_weight,
                    left_weight,
                    base_error,
                    base_components,
                ):
                    accepted_scale = candidate_scale
                    break
            safe_scale[tick] = min(safe_scale[tick], accepted_scale)
            if accepted_scale <= 0.0:
                continue
            applied = accepted_scale * joint_delta
            accumulated_delta[tick] += applied
            candidate = q_base.copy()
            candidate[groups.recovery_indices] += applied
            model.forward(candidate)
            recovered[tick] = model.get_qpos()
        qpos = recovered
    return qpos, safe_scale, accumulated_delta


def _run_ground_passes(
    model: MujocoModel,
    solver: BodyPositionIKSolver,
    bodies: _FpaBodies,
    groups: _FpaJointGroups,
    profile: FpaProfile,
    qpos_input: FloatArray,
    right_contact: FloatArray,
    left_contact: FloatArray,
    dt: float,
) -> _GroundState:
    qpos = qpos_input.copy()
    before_ground = qpos.copy()
    right_pre, left_pre = _ground_distances(model, bodies, qpos)
    right_need = np.where(right_contact >= 0.5, np.maximum(0.001 - right_pre, 0.0), 0.0)
    left_need = np.where(left_contact >= 0.5, np.maximum(0.001 - left_pre, 0.0), 0.0)
    correction_raw = np.clip(
        np.maximum(right_need, left_need),
        0.0,
        profile.ground_geometry_max_correction,
    )
    correction = slew_limited_upper_envelope(
        correction_raw,
        max_step=profile.ground_geometry_ramp_speed * dt,
        max_value=profile.ground_geometry_max_correction,
    )
    qpos[:, 2] += correction

    # Distribution profiles intentionally disable the optional main
    # post-ground joint recovery branch from the research cell.
    if profile.post_ground_recovery_passes != 0:
        raise ValueError("The supported FPA profiles must disable main recovery")
    micro_lift: FloatArray = np.zeros(len(qpos), dtype=np.float64)
    if profile.post_ground_micro_lift_max > 0.0:
        right_now, left_now = _ground_distances(model, bodies, qpos)
        right_weight = (
            np.ones(len(qpos), dtype=np.float64)
            if profile.post_ground_micro_lift_include_swing_feet
            else right_contact
        )
        left_weight = (
            np.ones(len(qpos), dtype=np.float64)
            if profile.post_ground_micro_lift_include_swing_feet
            else left_contact
        )
        residual_need = np.maximum(
            np.where(
                right_weight >= 0.5,
                profile.ground_geometry_clearance - right_now,
                0.0,
            ),
            np.where(
                left_weight >= 0.5,
                profile.ground_geometry_clearance - left_now,
                0.0,
            ),
        )
        speed = (
            profile.post_ground_micro_lift_speed
            if profile.post_ground_micro_lift_speed > 0.0
            else profile.ground_geometry_ramp_speed
        )
        micro_lift = slew_limited_upper_envelope(
            residual_need,
            max_step=speed * dt,
            max_value=profile.post_ground_micro_lift_max,
        )
        qpos[:, 2] += micro_lift

    dual_support_lower: FloatArray = np.zeros(len(qpos), dtype=np.float64)
    if profile.post_ground_dual_support_lower_max > 0.0:
        right_now, left_now = _ground_distances(model, bodies, qpos)
        support_weight = (
            np.maximum(right_contact, left_contact)
            if profile.post_ground_root_lower_support_mode == "any"
            else np.minimum(right_contact, left_contact)
        )
        safe_lower_raw = np.maximum(np.minimum(right_now, left_now) - 0.001, 0.0) * support_weight
        dual_support_lower = slew_limited_lower_envelope(
            safe_lower_raw,
            max_step=profile.post_ground_dual_support_lower_speed * dt,
            max_value=profile.post_ground_dual_support_lower_max,
        )
        qpos[:, 2] -= dual_support_lower

    qpos, dual_safe_scale, dual_joint_delta = _run_dual_recovery(
        model,
        solver,
        bodies,
        groups,
        profile,
        qpos,
        right_contact,
        left_contact,
        dual_support_lower,
        dt,
    )
    right_post, left_post = _ground_distances(model, bodies, qpos)
    return _GroundState(
        qpos=qpos,
        qpos_before_ground=before_ground,
        geometry_correction=correction,
        right_distance_pre=right_pre,
        left_distance_pre=left_pre,
        right_distance_post=right_post,
        left_distance_post=left_post,
        micro_lift=micro_lift,
        dual_support_lower=dual_support_lower,
        dual_recovery_safe_scale=dual_safe_scale,
        dual_recovery_joint_delta=dual_joint_delta,
    )


def solve_fpa(
    targets: FpaTargetsResult,
    trajectories: TargetTrajectoriesResult,
    contacts: ContactSchedule,
    *,
    robot_id: str,
    fps: float,
    backend: BackendPreference | BackendSelection = "python",
    source_provider: Literal["kimodo", "gem-x"] = "kimodo",
    left_ankle_target_rotations: ArrayLike | None = None,
    right_ankle_target_rotations: ArrayLike | None = None,
) -> FpaIkResult:
    if not (
        targets.robot_id == trajectories.robot_id == robot_id
        and contacts.frame_count == len(targets.seconds)
    ):
        raise ValueError("Stage 7 inputs do not describe the same robot/motion")
    if not (
        np.array_equal(targets.seconds, trajectories.seconds)
        and np.array_equal(targets.seconds, contacts.seconds)
    ):
        raise ValueError("Stage 7 timestamps do not match")
    if float(fps) != targets.fps or float(fps) != trajectories.fps:
        raise ValueError("Stage 7 fps does not match its inputs")
    if len(targets.seconds) < 2:
        raise ValueError("Stage 7 requires at least two frames")
    dt = float(targets.seconds[1] - targets.seconds[0])
    backend_selection = resolve_backend(backend)
    profile = get_fpa_profile(robot_id, source_provider=source_provider)
    dmr_profile = get_dmr_profile(robot_id, source_provider=source_provider)
    right_orientation_body: str | None = None
    left_orientation_body: str | None = None
    right_ankle_rotations: FloatArray | None = None
    left_ankle_rotations: FloatArray | None = None
    if profile.use_profile_ankle_orientation:
        if left_ankle_target_rotations is None or right_ankle_target_rotations is None:
            raise ValueError("This source profile requires DMR ankle target rotations")
        left_ankle_rotations = np.asarray(left_ankle_target_rotations, dtype=np.float64)
        right_ankle_rotations = np.asarray(right_ankle_target_rotations, dtype=np.float64)
        expected = (len(targets.seconds), 3, 3)
        if (
            left_ankle_rotations.shape != expected
            or right_ankle_rotations.shape != expected
            or not np.isfinite(left_ankle_rotations).all()
            or not np.isfinite(right_ankle_rotations).all()
        ):
            raise ValueError(f"DMR ankle target rotations must be finite with shape {expected}")
        left_key = dmr_profile.left_ankle_orientation_joi_key or (
            "la" if dmr_profile.ankle_orientation_stage == "primary" else "lf"
        )
        right_key = dmr_profile.right_ankle_orientation_joi_key or (
            "ra" if dmr_profile.ankle_orientation_stage == "primary" else "rf"
        )
        left_orientation_body = dmr_profile.joi_bodies[left_key]
        right_orientation_body = dmr_profile.joi_bodies[right_key]
    model = MujocoModel.from_robot(robot_id)
    ik_model = MujocoModel.from_robot(robot_id)
    solver = _make_fpa_solver(ik_model, backend_selection)
    bodies = _fpa_bodies(robot_id)
    groups = _fpa_joint_groups(model, profile)
    primary = _run_primary_ik(
        model,
        solver,
        bodies,
        groups,
        targets,
        contacts,
        profile,
        right_orientation_body=right_orientation_body,
        left_orientation_body=left_orientation_body,
        right_ankle_rotations=right_ankle_rotations,
        left_ankle_rotations=left_ankle_rotations,
    )
    base = _run_base_recovery(
        model,
        solver,
        bodies,
        groups,
        profile,
        targets,
        contacts,
        primary,
        dt,
    )
    ground = _run_ground_passes(
        model,
        solver,
        bodies,
        groups,
        profile,
        base.qpos,
        primary.right_contact_weight,
        primary.left_contact_weight,
        dt,
    )
    return FpaIkResult(
        robot_id=robot_id,
        fps=float(fps),
        seconds=targets.seconds,
        qpos=ground.qpos,
        qpos_before_base_smoothing=base.qpos_before_smoothing,
        qpos_after_base_smoothing_before_ground=ground.qpos_before_ground,
        qpos_ara_before_adaptive_smoothing=base.qpos_ara_before_smoothing,
        qpos_ara_after_adaptive_smoothing=base.qpos_ara_after_smoothing,
        ik_error=primary.error,
        ik_error_recovery=base.recovery_error,
        right_contact_weight=primary.right_contact_weight,
        left_contact_weight=primary.left_contact_weight,
        right_control_weight=primary.right_control_weight,
        left_control_weight=primary.left_control_weight,
        right_ik_weight=primary.right_control_weight,
        left_ik_weight=primary.left_control_weight,
        joint_correction_raw=base.joint_correction_raw,
        joint_correction_smooth=base.joint_correction_smooth,
        root_z_correction=base.root_z_correction,
        base_smoothing_delta=base.base_smoothing_delta,
        ara_smoothing_weight=base.ara_smoothing_weight,
        ara_smoothing_delta=base.ara_smoothing_delta,
        ara_adaptive_weight_xyz=base.ara_adaptive_weight_xyz,
        ara_flight_guard_mask=base.ara_flight_guard_mask,
        ground_geometry_correction=ground.geometry_correction,
        right_ground_distance_pre=ground.right_distance_pre,
        left_ground_distance_pre=ground.left_distance_pre,
        right_ground_distance_post=ground.right_distance_post,
        left_ground_distance_post=ground.left_distance_post,
        post_ground_micro_lift=ground.micro_lift,
        post_ground_dual_support_lower=ground.dual_support_lower,
        post_ground_dual_recovery_safe_scale=ground.dual_recovery_safe_scale,
        post_ground_dual_recovery_joint_delta=ground.dual_recovery_joint_delta,
        solve_records=base.solve_records,
    )


def run_fpa(
    qpos_stage3: ArrayLike,
    trajectories: TargetTrajectoriesResult,
    ara: AraResultLike,
    contacts: ContactSchedule,
    *,
    robot_id: str,
    fps: float,
    backend: BackendPreference | BackendSelection = "python",
    source_provider: Literal["kimodo", "gem-x"] = "kimodo",
    left_ankle_target_rotations: ArrayLike | None = None,
    right_ankle_target_rotations: ArrayLike | None = None,
) -> FpaResult:
    targets = build_fpa_targets(
        qpos_stage3,
        trajectories,
        ara,
        contacts,
        robot_id=robot_id,
        fps=fps,
        source_provider=source_provider,
    )
    return FpaResult(
        targets=targets,
        ik=solve_fpa(
            targets,
            trajectories,
            contacts,
            robot_id=robot_id,
            fps=fps,
            backend=backend,
            source_provider=source_provider,
            left_ankle_target_rotations=left_ankle_target_rotations,
            right_ankle_target_rotations=right_ankle_target_rotations,
        ),
    )


__all__ = [
    "FPA_IK_SOLVE_LABELS",
    "FPA_TARGET_SOLVE_LABELS",
    "AraResultLike",
    "FpaIkResult",
    "FpaResult",
    "FpaSolveRecord",
    "FpaTargetsResult",
    "build_fpa_targets",
    "run_fpa",
    "solve_fpa",
]
