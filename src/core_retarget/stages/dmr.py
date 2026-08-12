"""Profile-driven direct motion retargeting (DMR).

Robot-specific constants and capability switches live in immutable profiles;
this module never branches on a robot identifier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d, median_filter  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.kinematics import position, rotation, transform, unit_vector
from core_retarget.motion import SomaJoiTrajectory, SomaMotion, extract_soma_joi
from core_retarget.motion.source_frame import canonical_soma_pelvis_rotations
from core_retarget.mujoco.ik import BodyPositionIKSolver
from core_retarget.mujoco.model import MujocoModel
from core_retarget.mujoco.robot_kinematics import NeutralRobotGeometry, derive_neutral_geometry
from core_retarget.native import BackendPreference, BackendSelection, resolve_backend
from core_retarget.robots.profiles import DmrProfile, IkSolverProfile, get_dmr_profile

DmrProgress = Callable[[int, int, float], None]


def _readonly(array: NDArray[np.generic], *, dtype: np.dtype[np.generic]) -> NDArray[np.generic]:
    value = np.array(array, dtype=dtype, copy=True, order="C")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class DmrResult:
    """Numerical output of the DMR stage before collision refinement."""

    robot_id: str
    fps: float
    seconds: NDArray[np.float64]
    qpos: NDArray[np.float32]
    base_target_rotations: NDArray[np.float64]
    base_effective_target_rotations: NDArray[np.float64]
    torso_target_rotations: NDArray[np.float64] | None
    pelvis_stabilization_weight: NDArray[np.float64]
    pelvis_low_motion_weight: NDArray[np.float64]
    pelvis_fast_motion_weight: NDArray[np.float64]
    pelvis_tilt_blend: NDArray[np.float64]
    pelvis_upright_blend: NDArray[np.float64]
    pelvis_upright_pose_weight: NDArray[np.float64]
    pelvis_orientation_weight: NDArray[np.float64]
    pelvis_decoupled_solve_blend: NDArray[np.float64]
    trunk_position_blend: NDArray[np.float64]
    source_provider: Literal["kimodo", "gem-x"] = "kimodo"
    left_ankle_target_rotations: NDArray[np.float64] | None = None
    right_ankle_target_rotations: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        frame_count = len(self.seconds)
        arrays_1d = (
            "pelvis_stabilization_weight",
            "pelvis_low_motion_weight",
            "pelvis_fast_motion_weight",
            "pelvis_tilt_blend",
            "pelvis_upright_blend",
            "pelvis_upright_pose_weight",
            "pelvis_orientation_weight",
            "pelvis_decoupled_solve_blend",
            "trunk_position_blend",
        )
        object.__setattr__(self, "seconds", _readonly(self.seconds, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "qpos", _readonly(self.qpos, dtype=np.dtype(np.float32)))
        for field_name in ("base_target_rotations", "base_effective_target_rotations"):
            object.__setattr__(
                self,
                field_name,
                _readonly(getattr(self, field_name), dtype=np.dtype(np.float64)),
            )
        if self.torso_target_rotations is not None:
            object.__setattr__(
                self,
                "torso_target_rotations",
                _readonly(self.torso_target_rotations, dtype=np.dtype(np.float64)),
            )
        for field_name in (
            "left_ankle_target_rotations",
            "right_ankle_target_rotations",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _readonly(value, dtype=np.dtype(np.float64)),
                )
        for field_name in arrays_1d:
            object.__setattr__(
                self,
                field_name,
                _readonly(getattr(self, field_name), dtype=np.dtype(np.float64)),
            )

        if self.seconds.shape != (frame_count,):
            raise ValueError("DMR seconds must be one-dimensional.")
        if self.qpos.ndim != 2 or self.qpos.shape[0] != frame_count:
            raise ValueError("DMR qpos must have shape (frames, nq).")
        expected_rotation_shape = (frame_count, 3, 3)
        if self.base_target_rotations.shape != expected_rotation_shape:
            raise ValueError("DMR base target rotations have an invalid shape.")
        if self.base_effective_target_rotations.shape != expected_rotation_shape:
            raise ValueError("DMR effective base rotations have an invalid shape.")
        if (
            self.torso_target_rotations is not None
            and self.torso_target_rotations.shape != expected_rotation_shape
        ):
            raise ValueError("DMR torso target rotations have an invalid shape.")
        if self.source_provider not in {"kimodo", "gem-x"}:
            raise ValueError("DMR source_provider must be 'kimodo' or 'gem-x'.")
        if (self.left_ankle_target_rotations is None) != (
            self.right_ankle_target_rotations is None
        ):
            raise ValueError("DMR ankle target rotations must be supplied as a pair.")
        for field_name in (
            "left_ankle_target_rotations",
            "right_ankle_target_rotations",
        ):
            value = getattr(self, field_name)
            if value is not None and value.shape != expected_rotation_shape:
                raise ValueError(f"DMR field {field_name} has an invalid shape.")
        for field_name in arrays_1d:
            if getattr(self, field_name).shape != (frame_count,):
                raise ValueError(f"DMR field {field_name} has an invalid shape.")

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 2 archive arrays."""

        arrays: dict[str, NDArray[np.generic]] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_dmr_array": self.qpos,
            "R_base_trgt_smt_array": self.base_target_rotations,
            "R_base_effective_trgt_array": self.base_effective_target_rotations,
        }
        if self.torso_target_rotations is not None:
            arrays["R_torso_trgt_smt_array"] = self.torso_target_rotations
        arrays.update(
            {
                "pelvis_stabilization_weight_array": self.pelvis_stabilization_weight,
                "pelvis_low_motion_weight_array": self.pelvis_low_motion_weight,
                "pelvis_fast_motion_weight_array": self.pelvis_fast_motion_weight,
                "pelvis_tilt_blend_array": self.pelvis_tilt_blend,
                "pelvis_upright_blend_array": self.pelvis_upright_blend,
                "pelvis_upright_pose_weight_array": self.pelvis_upright_pose_weight,
                "pelvis_orientation_weight_array": self.pelvis_orientation_weight,
                "pelvis_decoupled_solve_blend_array": self.pelvis_decoupled_solve_blend,
                "trunk_position_blend_array": self.trunk_position_blend,
            }
        )
        return arrays


@dataclass(frozen=True)
class _JointGroups:
    body: tuple[str, ...]
    wrist: tuple[str, ...]
    waist: tuple[str, ...]
    ankle: tuple[str, ...]
    toe: tuple[str, ...]


def build_base_orientation_targets(
    motion: SomaMotion,
    *,
    smooth_time: float,
    smoothing_mode: str = "rotvec_legacy",
) -> NDArray[np.float64]:
    """Build smoothed canonical pelvis rotation targets."""

    source_rotations = canonical_soma_pelvis_rotations(motion)
    return _smooth_rotation_sequence(
        source_rotations,
        smooth_time=smooth_time,
        dt=1.0 / motion.fps,
        mode=smoothing_mode,
    )


def _frame_transforms(joi: SomaJoiTrajectory, tick: int) -> Mapping[str, NDArray[np.float64]]:
    return {name: joi.transforms[tick, index] for index, name in enumerate(joi.names)}


def _resolve_semantic_joi_anchors(
    model: MujocoModel,
    profile: DmrProfile,
) -> dict[str, NDArray[np.float64]]:
    """Resolve profile landmarks as neutral body-local points."""

    anchors: dict[str, NDArray[np.float64]] = {}
    for key, references in profile.joi_anchor_reference_keys.items():
        body_transform = model.get_body_transform(profile.joi_bodies[key])
        reference_position = np.mean(
            [
                position(model.get_body_transform(profile.joi_bodies[reference]))
                for reference in references
            ],
            axis=0,
        )
        anchors[key] = np.matmul(
            rotation(body_transform).T,
            reference_position - position(body_transform),
        )
    return anchors


def _semantic_joi_position(
    model: MujocoModel,
    profile: DmrProfile,
    key: str,
    semantic_anchors: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    body_transform = model.get_body_transform(profile.joi_bodies[key])
    local_position = semantic_anchors.get(key)
    if local_position is None:
        return position(body_transform)
    return np.asarray(
        position(body_transform) + np.matmul(rotation(body_transform), local_position),
        dtype=np.float64,
    )


def _semantic_joi_transform(
    model: MujocoModel,
    profile: DmrProfile,
    key: str,
    semantic_anchors: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    body_transform = model.get_body_transform(profile.joi_bodies[key])
    if key not in semantic_anchors:
        return body_transform
    return transform(
        _semantic_joi_position(model, profile, key, semantic_anchors),
        rotation(body_transform),
    )


def _matches_any_token(name: str, tokens: Sequence[str]) -> bool:
    lower_name = str(name).lower()
    return any(str(token).lower() in lower_name for token in tokens)


def _joint_groups(model: MujocoModel, profile: DmrProfile) -> _JointGroups:
    names = tuple(model.rev_pri_joint_names)
    wrist = tuple(name for name in names if _matches_any_token(name, profile.wrist_joint_tokens))
    waist = tuple(name for name in names if _matches_any_token(name, profile.waist_joint_tokens))
    ankle = tuple(name for name in names if _matches_any_token(name, profile.ankle_joint_tokens))
    toe = tuple(name for name in names if _matches_any_token(name, profile.toe_joint_tokens))
    body = tuple(
        name
        for name in names
        if name not in wrist
        and (not profile.exclude_waist_from_primary_dmr or name not in waist)
        and (profile.optimize_toe_dmr or name not in toe)
    )
    return _JointGroups(body=body, wrist=wrist, waist=waist, ankle=ankle, toe=toe)


def _validate_active_joint_groups(profile: DmrProfile, groups: _JointGroups) -> None:
    """Reject active profile stages that resolved no controllable joints."""

    if profile.torso_orientation_stage == "post" and not groups.waist:
        raise ConfigurationError(
            f"Robot {profile.robot_id!r} activates the torso post solver, but "
            "waist_joint_tokens matched no model joints."
        )
    if profile.ankle_orientation_stage == "post" and not groups.ankle:
        raise ConfigurationError(
            f"Robot {profile.robot_id!r} activates the ankle post solver, but "
            "ankle_joint_tokens matched no model joints."
        )
    if (
        profile.hand_orientation_enabled
        and profile.hand_orientation_axis_length > 0.0
        and not groups.wrist
    ):
        raise ConfigurationError(
            f"Robot {profile.robot_id!r} activates hand orientation, but "
            "wrist_joint_tokens matched no model joints."
        )


def _confidence_array(
    values: ArrayLike | None,
    frame_count: int,
    name: str,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) != int(frame_count):
        raise ValueError(f"{name} length mismatch: expected {frame_count}, got {len(array)}")
    return np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)


def _smooth_falloff(
    values: ArrayLike,
    low: float,
    high: float,
) -> NDArray[np.float64]:
    values_array = np.asarray(values, dtype=np.float64)
    phase = np.clip((values_array - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    smoothstep = phase * phase * (3.0 - 2.0 * phase)
    return np.asarray(1.0 - smoothstep, dtype=np.float64)


def _slew_limited_confidence(
    values: NDArray[np.float64],
    *,
    dt: float,
    transition_time: float,
) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if len(clipped) <= 1 or float(transition_time) <= 0.0:
        return clipped.copy()
    max_step = min(1.0, max(float(dt), 1e-12) / max(float(transition_time), 1e-12))
    output = clipped.copy()
    for tick in range(1, len(output)):
        output[tick] = np.clip(
            output[tick], output[tick - 1] - max_step, output[tick - 1] + max_step
        )
    return output


def _source_body_scale(source: Sequence[Mapping[str, NDArray[np.float64]]]) -> float:
    scales: list[float] = []
    for transforms in source:
        points = {
            key: position(transforms[key])
            for key in (
                "base",
                "spine",
                "neck",
                "rp",
                "rk",
                "ra",
                "rtoe",
                "lp",
                "lk",
                "la",
                "ltoe",
            )
        }
        trunk = np.linalg.norm(points["spine"] - points["base"]) + np.linalg.norm(
            points["neck"] - points["spine"]
        )
        right_leg = sum(
            np.linalg.norm(points[end] - points[start])
            for start, end in (
                ("base", "rp"),
                ("rp", "rk"),
                ("rk", "ra"),
                ("ra", "rtoe"),
            )
        )
        left_leg = sum(
            np.linalg.norm(points[end] - points[start])
            for start, end in (
                ("base", "lp"),
                ("lp", "lk"),
                ("lk", "la"),
                ("la", "ltoe"),
            )
        )
        scales.append(float(trunk + 0.5 * (right_leg + left_leg)))
    scale = float(np.median(scales)) if scales else 1.0
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def _contact_motion_stability(
    *,
    profile: DmrProfile,
    source: Sequence[Mapping[str, NDArray[np.float64]]],
    source_base_rotations: NDArray[np.float64],
    dt: float,
    left_contact_confidence: ArrayLike | None,
    right_contact_confidence: ArrayLike | None,
) -> dict[str, NDArray[np.float64] | float]:
    frame_count = len(source)
    zeros = np.zeros(frame_count, dtype=np.float64)
    left = _confidence_array(left_contact_confidence, frame_count, "left_contact_confidence")
    right = _confidence_array(right_contact_confidence, frame_count, "right_contact_confidence")
    if left is None or right is None or frame_count == 0:
        return {
            "weight": zeros,
            "double_support": zeros,
            "double_support_raw": zeros,
            "low_motion": zeros,
            "linear_speed_normalized": zeros,
            "angular_speed": zeros,
            "source_body_scale": 1.0,
        }

    source_position = np.asarray(
        [position(transforms["base"]) for transforms in source], dtype=np.float64
    )
    if frame_count > 1:
        linear_velocity = np.gradient(source_position, max(float(dt), 1e-12), axis=0, edge_order=1)
        linear_speed = np.linalg.norm(linear_velocity, axis=1)
    else:
        linear_speed = zeros.copy()
    source_scale = _source_body_scale(source)
    linear_speed_normalized = linear_speed / source_scale

    angular_speed = zeros.copy()
    if frame_count > 1:
        delta_rotation = np.matmul(
            np.swapaxes(source_base_rotations[:-1], 1, 2),
            source_base_rotations[1:],
        )
        step_speed = Rotation.from_matrix(delta_rotation).magnitude() / max(float(dt), 1e-12)
        angular_speed[0] = step_speed[0]
        angular_speed[-1] = step_speed[-1]
        if frame_count > 2:
            angular_speed[1:-1] = 0.5 * (step_speed[:-1] + step_speed[1:])

    activity_sigma = 0.5 * float(profile.pelvis_stabilization_smooth_time) / max(float(dt), 1e-12)
    if frame_count > 1 and activity_sigma > 1e-6:
        linear_speed_normalized = gaussian_filter1d(
            linear_speed_normalized, sigma=activity_sigma, mode="nearest"
        )
        angular_speed = gaussian_filter1d(angular_speed, sigma=activity_sigma, mode="nearest")

    low_linear = _smooth_falloff(
        linear_speed_normalized,
        profile.pelvis_stabilization_linear_speed_low,
        profile.pelvis_stabilization_linear_speed_high,
    )
    low_angular = _smooth_falloff(
        angular_speed,
        profile.pelvis_stabilization_angular_speed_low,
        profile.pelvis_stabilization_angular_speed_high,
    )
    double_support_raw = np.clip(left * right, 0.0, 1.0)
    double_support = _slew_limited_confidence(
        double_support_raw,
        dt=dt,
        transition_time=profile.pelvis_support_transition_time,
    )
    activity_weight = low_linear * low_angular
    gate_sigma = float(profile.pelvis_stabilization_smooth_time) / max(float(dt), 1e-12)
    if frame_count > 1 and gate_sigma > 1e-6:
        activity_weight = gaussian_filter1d(activity_weight, sigma=gate_sigma, mode="nearest")
    return {
        "weight": np.clip(double_support * activity_weight, 0.0, 1.0),
        "double_support": double_support,
        "double_support_raw": double_support_raw,
        "low_motion": np.clip(activity_weight, 0.0, 1.0),
        "linear_speed_normalized": linear_speed_normalized,
        "angular_speed": angular_speed,
        "source_body_scale": source_scale,
    }


def _stabilize_relative_tilt(
    relative_rotations: NDArray[np.float64],
    blend: NDArray[np.float64],
) -> NDArray[np.float64]:
    stabilized = np.empty_like(relative_rotations)
    for tick, (relative_rotation, alpha) in enumerate(zip(relative_rotations, blend, strict=False)):
        yaw = np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])
        yaw_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, yaw])).as_matrix()
        tilt_rotation = np.matmul(yaw_rotation.T, relative_rotation)
        tilt_rotvec = Rotation.from_matrix(tilt_rotation).as_rotvec()
        stabilized[tick] = np.matmul(
            yaw_rotation,
            Rotation.from_rotvec((1.0 - float(alpha)) * tilt_rotvec).as_matrix(),
        )
    return stabilized


def _heading_rotation_and_tilt(
    value: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    heading = Rotation.from_rotvec(np.array([0.0, 0.0, yaw])).as_matrix()
    return heading, np.matmul(heading.T, matrix)


def _minimal_rotation_between(
    vector_from: ArrayLike,
    vector_to: ArrayLike,
) -> NDArray[np.float64]:
    source = unit_vector(vector_from)
    target = unit_vector(vector_to)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm < 1e-10:
        if dot >= 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        axis = unit_vector(axis, fallback=(1.0, 0.0, 0.0))
        return np.asarray(Rotation.from_rotvec(np.pi * axis).as_matrix(), dtype=np.float64)
    axis = cross / cross_norm
    return np.asarray(
        Rotation.from_rotvec(np.arctan2(cross_norm, dot) * axis).as_matrix(),
        dtype=np.float64,
    )


def _smooth_rotation_sequence(
    rotations: ArrayLike,
    *,
    smooth_time: float,
    dt: float,
    mode: str = "rotvec_legacy",
) -> NDArray[np.float64]:
    values = np.asarray(rotations, dtype=np.float64)
    if len(values) <= 1 or smooth_time <= 0.0:
        return values.copy()
    sigma = max(float(smooth_time) / max(float(dt), 1e-12), 1e-6)
    if mode == "rotvec_legacy":
        reference = values[0]
        relative = np.matmul(reference.T[None, :, :], values)
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        rotvec_smooth = gaussian_filter1d(rotvec, sigma=sigma, axis=0, mode="nearest")
        relative_smooth = Rotation.from_rotvec(rotvec_smooth).as_matrix()
        return np.asarray(np.matmul(reference[None, :, :], relative_smooth), dtype=np.float64)
    if mode != "quaternion_continuous":
        raise ValueError(f"Unsupported rotation smoothing mode: {mode}")
    quaternions = Rotation.from_matrix(values).as_quat()
    for tick in range(1, len(quaternions)):
        if float(np.dot(quaternions[tick - 1], quaternions[tick])) < 0.0:
            quaternions[tick] *= -1.0
    smoothed = gaussian_filter1d(quaternions, sigma=sigma, axis=0, mode="nearest")
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Rotation smoothing produced a near-zero quaternion.")
    return np.asarray(Rotation.from_quat(smoothed / norms).as_matrix(), dtype=np.float64)


def _build_torso_orientation_targets(
    *,
    profile: DmrProfile,
    source: Sequence[Mapping[str, NDArray[np.float64]]],
    source_base_rotations: NDArray[np.float64],
    target_base_rotations: NDArray[np.float64],
    geometry: NeutralRobotGeometry,
    robot_torso_transform: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64] | None:
    if profile.torso_orientation_weight <= 0.0:
        return None
    source_tilt: list[NDArray[np.float64]] = []
    for transforms, base_rotation in zip(source, source_base_rotations, strict=False):
        source_up_world = unit_vector(position(transforms["neck"]) - position(transforms["spine"]))
        source_up_local = np.matmul(base_rotation.T, source_up_world)
        source_tilt.append(_minimal_rotation_between((0.0, 0.0, 1.0), source_up_local))
    tilt_array = np.asarray(source_tilt, dtype=np.float64)
    robot_relative = np.matmul(
        rotation(geometry.length_transforms["base"]).T,
        rotation(robot_torso_transform),
    )
    if profile.torso_orientation_reference_mode == "source_delta":
        source_transfer = np.matmul(tilt_array, tilt_array[0].T)
    else:
        source_transfer = tilt_array
    relative_target = np.matmul(source_transfer, robot_relative)
    relative_target = _smooth_rotation_sequence(
        relative_target,
        smooth_time=profile.torso_orientation_smooth_time,
        dt=dt,
        mode=profile.orientation_smoothing_mode,
    )
    return np.asarray(np.matmul(target_base_rotations, relative_target), dtype=np.float64)


def _build_ankle_orientation_targets(
    *,
    profile: DmrProfile,
    joi: SomaJoiTrajectory,
    geometry: NeutralRobotGeometry,
    dt: float,
    left_contact_confidence: ArrayLike | None = None,
    right_contact_confidence: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]] | None:
    if profile.ankle_orientation_mode == "none":
        return None
    left_key = profile.left_ankle_orientation_joi_key or (
        "la" if profile.ankle_orientation_stage == "primary" else "lf"
    )
    right_key = profile.right_ankle_orientation_joi_key or (
        "ra" if profile.ankle_orientation_stage == "primary" else "rf"
    )
    if profile.ankle_orientation_mode == "source_body":
        left = np.matmul(
            joi.rotations("la"),
            np.asarray(profile.left_ankle_local_offset, dtype=np.float64),
        )
        right = np.matmul(
            joi.rotations("ra"),
            np.asarray(profile.right_ankle_local_offset, dtype=np.float64),
        )
    else:
        left_source = _smooth_rotation_sequence(
            joi.rotations("la"),
            smooth_time=profile.ankle_orientation_smooth_time,
            dt=dt,
            mode=profile.orientation_smoothing_mode,
        )
        right_source = _smooth_rotation_sequence(
            joi.rotations("ra"),
            smooth_time=profile.ankle_orientation_smooth_time,
            dt=dt,
            mode=profile.orientation_smoothing_mode,
        )
        left_offset = np.matmul(left_source[0].T, rotation(geometry.length_transforms[left_key]))
        right_offset = np.matmul(right_source[0].T, rotation(geometry.length_transforms[right_key]))
        left = np.matmul(left_source, left_offset)
        right = np.matmul(right_source, right_offset)
    left = _flatten_contact_rotations(
        left,
        left_contact_confidence,
        strength=profile.ankle_contact_flatten_strength,
        smooth_time=profile.ankle_contact_flatten_smooth_time,
        dt=dt,
    )
    right = _flatten_contact_rotations(
        right,
        right_contact_confidence,
        strength=profile.ankle_contact_flatten_strength,
        smooth_time=profile.ankle_contact_flatten_smooth_time,
        dt=dt,
    )
    return {
        "left": np.asarray(left, dtype=np.float64),
        "right": np.asarray(right, dtype=np.float64),
    }


def _ground_aligned_rotation(value: ArrayLike) -> NDArray[np.float64]:
    """Keep a semantic sole heading while aligning its normal to world +Z."""

    source = np.asarray(value, dtype=np.float64).reshape(3, 3)
    z_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    x_axis: NDArray[np.float64] = np.asarray(source[:, 0], dtype=np.float64).copy()
    x_axis[2] = 0.0
    if np.linalg.norm(x_axis) < 1e-8:
        y_axis = source[:, 1].copy()
        y_axis[2] = 0.0
        y_axis = unit_vector(y_axis, fallback=(0.0, 1.0, 0.0))
        x_axis = np.asarray(np.cross(y_axis, z_axis), dtype=np.float64)
    x_axis = unit_vector(x_axis, fallback=(1.0, 0.0, 0.0))
    y_axis = unit_vector(np.cross(z_axis, x_axis), fallback=(0.0, 1.0, 0.0))
    x_axis = unit_vector(np.cross(y_axis, z_axis), fallback=(1.0, 0.0, 0.0))
    return np.column_stack((x_axis, y_axis, z_axis))


def _flatten_contact_rotations(
    rotations: ArrayLike,
    contact_confidence: ArrayLike | None,
    *,
    strength: float,
    smooth_time: float,
    dt: float,
) -> NDArray[np.float64]:
    output = np.asarray(rotations, dtype=np.float64).copy()
    if contact_confidence is None or strength <= 0.0:
        return output
    confidence: NDArray[np.float64] = np.asarray(contact_confidence, dtype=np.float64).reshape(-1)
    if confidence.shape != (len(output),):
        raise ValueError("Ankle contact confidence does not match the motion frame count.")
    weights = np.asarray(np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0), dtype=np.float64)
    if len(weights) > 1 and smooth_time > 0.0:
        sigma = max(float(smooth_time) / max(float(dt), 1e-12), 1e-6)
        weights = np.asarray(
            gaussian_filter1d(weights, sigma=sigma, mode="nearest"), dtype=np.float64
        )
    for tick, alpha in enumerate(np.clip(float(strength) * weights, 0.0, 1.0)):
        if alpha <= 1e-10:
            continue
        floor_rotation = _ground_aligned_rotation(output[tick])
        delta = output[tick].T @ floor_rotation
        rotvec = Rotation.from_matrix(delta).as_rotvec()
        output[tick] = output[tick] @ Rotation.from_rotvec(float(alpha) * rotvec).as_matrix()
    return output


def _robot_neutral_delta_vector(
    robot_local_vector: NDArray[np.float64],
    source_reference_local: NDArray[np.float64],
    source_current_local: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer only a source direction change onto robot-neutral geometry."""

    return np.asarray(
        np.matmul(
            _minimal_rotation_between(source_reference_local, source_current_local),
            np.asarray(robot_local_vector, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def _body_targets(
    source: Mapping[str, NDArray[np.float64]],
    geometry: NeutralRobotGeometry,
    *,
    profile: DmrProfile,
    effective_base_rotation: NDArray[np.float64],
    trunk_blend: float,
    source_base_rotation: NDArray[np.float64],
    source_spine_reference_local: NDArray[np.float64],
    source_neck_reference_local: NDArray[np.float64],
    robot_spine_local: NDArray[np.float64],
    robot_neck_local: NDArray[np.float64],
) -> tuple[tuple[str, NDArray[np.float64]], ...]:
    lengths = geometry.link_lengths
    p_base = position(source["base"])
    source_spine_direction = unit_vector(position(source["spine"]) - position(source["base"]))
    source_neck_direction = unit_vector(position(source["neck"]) - position(source["spine"]))
    p_spine_world = p_base + lengths["base_spine"] * source_spine_direction
    p_neck_world = p_spine_world + lengths["spine_neck"] * source_neck_direction

    alpha = float(np.clip(trunk_blend, 0.0, 1.0))
    if alpha > 0.0 and profile.trunk_position_mode == "robot_bind_local":
        base_rotation = rotation(geometry.body_transforms["base"])
        base_position = position(geometry.body_transforms["base"])
        robot_bind_spine_local = np.matmul(
            base_rotation.T,
            position(geometry.body_transforms["spine"]) - base_position,
        )
        robot_bind_neck_local = np.matmul(
            base_rotation.T,
            position(geometry.body_transforms["neck"]) - base_position,
        )
        p_spine_local = p_base + np.matmul(effective_base_rotation, robot_bind_spine_local)
        p_neck_local = p_base + np.matmul(effective_base_rotation, robot_bind_neck_local)
        p_spine = (1.0 - alpha) * p_spine_world + alpha * p_spine_local
        p_neck = (1.0 - alpha) * p_neck_world + alpha * p_neck_local
    elif alpha > 0.0 and profile.trunk_position_mode == "robot_neutral_delta":
        source_spine_local = np.matmul(source_base_rotation.T, source_spine_direction)
        source_neck_local = np.matmul(source_base_rotation.T, source_neck_direction)
        robot_spine_target = _robot_neutral_delta_vector(
            robot_spine_local,
            source_spine_reference_local,
            source_spine_local,
        )
        robot_neck_segment_target = _robot_neutral_delta_vector(
            robot_neck_local - robot_spine_local,
            source_neck_reference_local,
            source_neck_local,
        )
        p_spine_local = p_base + np.matmul(
            effective_base_rotation,
            robot_spine_target,
        )
        p_neck_local = p_spine_local + np.matmul(
            effective_base_rotation,
            robot_neck_segment_target,
        )
        p_spine = (1.0 - alpha) * p_spine_world + alpha * p_spine_local
        p_neck = (1.0 - alpha) * p_neck_world + alpha * p_neck_local
    else:
        p_spine = p_spine_world
        p_neck = p_neck_world

    p_rs = p_neck + lengths["neck_rs"] * unit_vector(
        position(source["rs"]) - position(source["neck"])
    )
    p_re = p_rs + lengths["rs_re"] * unit_vector(position(source["re"]) - position(source["rs"]))
    p_rw = p_re + lengths["re_rw"] * unit_vector(position(source["rw"]) - position(source["re"]))
    p_ls = p_neck + lengths["neck_ls"] * unit_vector(
        position(source["ls"]) - position(source["neck"])
    )
    p_le = p_ls + lengths["ls_le"] * unit_vector(position(source["le"]) - position(source["ls"]))
    p_lw = p_le + lengths["le_lw"] * unit_vector(position(source["lw"]) - position(source["le"]))

    p_rp = p_base + lengths["base_rp"] * unit_vector(
        position(source["rp"]) - position(source["base"])
    )
    p_rk = p_rp + lengths["rp_rk"] * unit_vector(position(source["rk"]) - position(source["rp"]))
    p_ra = p_rk + lengths["rk_ra"] * unit_vector(position(source["ra"]) - position(source["rk"]))
    p_rt = p_ra + lengths["ra_rt"] * unit_vector(position(source["rtoe"]) - position(source["ra"]))
    p_lp = p_base + lengths["base_lp"] * unit_vector(
        position(source["lp"]) - position(source["base"])
    )
    p_lk = p_lp + lengths["lp_lk"] * unit_vector(position(source["lk"]) - position(source["lp"]))
    p_la = p_lk + lengths["lk_la"] * unit_vector(position(source["la"]) - position(source["lk"]))
    p_lt = p_la + lengths["la_lt"] * unit_vector(position(source["ltoe"]) - position(source["la"]))
    return (
        ("base", p_base),
        ("spine", p_spine),
        ("rs", p_rs),
        ("re", p_re),
        ("rw", p_rw),
        ("ls", p_ls),
        ("le", p_le),
        ("lw", p_lw),
        ("rp", p_rp),
        ("rk", p_rk),
        ("ra", p_ra),
        ("rt", p_rt),
        ("lp", p_lp),
        ("lk", p_lk),
        ("la", p_la),
        ("lt", p_lt),
    )


def _add_position_targets(
    solver: BodyPositionIKSolver,
    model: MujocoModel,
    profile: DmrProfile,
    targets: tuple[tuple[str, NDArray[np.float64]], ...],
    semantic_anchors: Mapping[str, NDArray[np.float64]],
) -> None:
    for key, target_position in targets:
        body_name = profile.joi_bodies[key]
        if key in semantic_anchors:
            current_position = _semantic_joi_position(
                model,
                profile,
                key,
                semantic_anchors,
            )
        else:
            # Preserve the established arithmetic for profiles that target a
            # mapped body origin directly.
            current_position = position(model.get_body_transform(body_name))
        solver.add_target(
            body_name,
            current_position,
            target_position,
        )


def _add_primary_ankle_targets(
    solver: BodyPositionIKSolver,
    model: MujocoModel,
    profile: DmrProfile,
    target_positions: Mapping[str, NDArray[np.float64]],
    target_rotations: Mapping[str, NDArray[np.float64]],
) -> None:
    for side, key in (("right", "ra"), ("left", "la")):
        body_name = profile.joi_bodies[key]
        current = model.get_body_transform(body_name)
        current_position = position(current)
        current_rotation = rotation(current)
        for axis in profile.ankle_orientation_axes:
            solver.add_target(
                body_name,
                current_position
                + profile.ankle_orientation_axis_length * current_rotation[:, axis],
                target_positions[key]
                + profile.ankle_orientation_axis_length * target_rotations[side][:, axis],
            )


def _make_solver(
    model: MujocoModel,
    profile_solver: IkSolverProfile,
    backend: BackendPreference | BackendSelection = "python",
) -> BodyPositionIKSolver:
    return BodyPositionIKSolver(
        model,
        max_iterations=profile_solver.max_iterations,
        revolute_step=profile_solver.revolute_step,
        revolute_update_limit=profile_solver.revolute_update_limit,
        damping=profile_solver.damping,
        joint_limit_probe=profile_solver.joint_limit_probe,
        home=model.get_qpos(model.rev_pri_joint_names),
        nullspace_gain=0.0,
        # The native backend uses scalar accumulation and Gaussian-elimination
        # order. Matching it here is necessary at the
        # float32 Stage 2 boundary: one-ULP differences can select a different
        # signed-collision branch in Stage 3.
        reference_ordered=True,
        backend=backend,
    )


def _resolve_contact_confidences(
    motion: SomaMotion,
    joi: SomaJoiTrajectory,
    *,
    left: ArrayLike | None,
    right: ArrayLike | None,
    source_provider: Literal["kimodo", "gem-x"],
) -> tuple[ArrayLike | None, ArrayLike | None]:
    if (left is None) != (right is None):
        raise ValueError(
            "left_contact_confidence and right_contact_confidence must be supplied together."
        )
    if left is not None:
        return left, right
    from core_retarget.motion.contacts import build_contact_schedule

    if source_provider == "gem-x" and motion.foot_contacts is not None:
        encoded = np.asarray(motion.foot_contacts, dtype=np.bool_)
        if encoded.ndim != 2 or encoded.shape[1] < 5:
            raise MotionValidationError(
                "GEM-X in-memory contacts must use the normalized six-channel layout."
            )
        schedule = build_contact_schedule(
            motion,
            source_joi=joi,
            left_source_labels=encoded[:, 1],
            right_source_labels=encoded[:, 4],
            source_contact_name="gemx_fused_toebase_contacts+time_varying_floor",
            floor_distance_threshold=0.03,
            maximum_contact_bridge_time=0.30,
        )
    else:
        schedule = build_contact_schedule(motion, source_joi=joi)
    return schedule.left_confidence, schedule.right_confidence


def run_dmr(
    motion: SomaMotion,
    *,
    robot_id: str,
    source_joi: SomaJoiTrajectory | None = None,
    left_contact_confidence: ArrayLike | None = None,
    right_contact_confidence: ArrayLike | None = None,
    progress: DmrProgress | None = None,
    backend: BackendPreference | BackendSelection = "python",
    source_provider: Literal["kimodo", "gem-x"] = "kimodo",
) -> DmrResult:
    """Retarget one SOMA motion with the selected verified robot profile."""

    profile = get_dmr_profile(robot_id, source_provider=source_provider)
    backend_selection = resolve_backend(backend)
    joi = extract_soma_joi(motion) if source_joi is None else source_joi
    if joi.frame_count != motion.frame_count:
        raise ValueError("SOMA motion and JOI trajectory frame counts do not match.")
    if not np.array_equal(joi.seconds, motion.seconds):
        raise MotionValidationError("SOMA motion and JOI trajectory timestamps do not match.")

    model = MujocoModel.from_robot(profile.robot_id)
    ik_model = MujocoModel.from_robot(profile.robot_id)
    model.reset()
    ik_model.reset()
    if int(model.model.nq) != profile.qpos_dim:
        raise ValueError(
            f"Profile qpos_dim={profile.qpos_dim} does not match model nq={model.model.nq}."
        )
    geometry = derive_neutral_geometry(
        model,
        profile.joi_bodies,
        link_length_base_reference=profile.link_length_base_reference,
    )
    semantic_anchors = _resolve_semantic_joi_anchors(model, profile)
    groups = _joint_groups(model, profile)
    _validate_active_joint_groups(profile, groups)
    body_solver = _make_solver(ik_model, profile.body_solver, backend_selection)
    hand_solver = _make_solver(ik_model, profile.hand_solver, backend_selection)
    torso_solver = (
        None
        if profile.torso_solver is None
        else _make_solver(ik_model, profile.torso_solver, backend_selection)
    )
    ankle_solver = (
        None
        if profile.ankle_solver is None
        else _make_solver(ik_model, profile.ankle_solver, backend_selection)
    )

    frame_count = motion.frame_count
    dt = 1.0 / motion.fps
    source = tuple(_frame_transforms(joi, tick) for tick in range(frame_count))
    source_base_rotations = canonical_soma_pelvis_rotations(motion)
    robot_base_rotation = rotation(geometry.body_transforms["base"])
    robot_base_anchor_position = _semantic_joi_position(
        model,
        profile,
        "base",
        semantic_anchors,
    )
    robot_spine_local = np.matmul(
        robot_base_rotation.T,
        position(geometry.body_transforms["spine"]) - robot_base_anchor_position,
    )
    robot_neck_local = np.matmul(
        robot_base_rotation.T,
        position(geometry.body_transforms["neck"]) - robot_base_anchor_position,
    )
    source_spine_reference_local = np.matmul(
        source_base_rotations[0].T,
        unit_vector(position(source[0]["spine"]) - position(source[0]["base"])),
    )
    source_neck_reference_local = np.matmul(
        source_base_rotations[0].T,
        unit_vector(position(source[0]["neck"]) - position(source[0]["spine"])),
    )
    stabilization_enabled = (
        profile.pelvis_stabilization_strength > 0.0
        or profile.pelvis_stabilization_orientation_weight > 0.0
        or profile.trunk_position_strength > 0.0
    )
    contacts_required = stabilization_enabled or profile.ankle_contact_flatten_strength > 0.0
    if contacts_required:
        left_contact_confidence, right_contact_confidence = _resolve_contact_confidences(
            motion,
            joi,
            left=left_contact_confidence,
            right=right_contact_confidence,
            source_provider=source_provider,
        )
    if stabilization_enabled:
        stability = _contact_motion_stability(
            profile=profile,
            source=source,
            source_base_rotations=source_base_rotations,
            dt=dt,
            left_contact_confidence=left_contact_confidence,
            right_contact_confidence=right_contact_confidence,
        )
    else:
        zeros = np.zeros(frame_count, dtype=np.float64)
        stability = {
            "weight": zeros,
            "double_support": zeros,
            "double_support_raw": zeros,
            "low_motion": zeros,
            "linear_speed_normalized": zeros,
            "angular_speed": zeros,
            "source_body_scale": _source_body_scale(source),
        }
    pelvis_stability_weight = np.asarray(stability["weight"], dtype=np.float64)
    pelvis_low_motion_weight = np.asarray(stability["low_motion"], dtype=np.float64)

    source_reference = source_base_rotations[0]
    if profile.orientation_smoothing_mode == "rotvec_legacy":
        # Preserve the frozen Kimodo arithmetic exactly. Reconstructing the
        # world rotation and then converting it back to this relative frame
        # introduces platform-dependent round-off at roughly 1e-9.
        base_relative = np.matmul(source_reference.T[None, :, :], source_base_rotations)
        base_rotvec = Rotation.from_matrix(base_relative).as_rotvec()
        base_sigma = max(
            profile.pelvis_orientation_smooth_time / max(float(dt), 1e-12),
            1e-6,
        )
        base_rotvec = gaussian_filter1d(
            base_rotvec,
            sigma=base_sigma,
            axis=0,
            mode="nearest",
        )
        base_relative_smooth = Rotation.from_rotvec(base_rotvec).as_matrix()
    else:
        base_smoothed = _smooth_rotation_sequence(
            source_base_rotations,
            smooth_time=profile.pelvis_orientation_smooth_time,
            dt=dt,
            mode=profile.orientation_smoothing_mode,
        )
        base_relative_smooth = np.matmul(source_reference.T[None, :, :], base_smoothed)
    if profile.pelvis_stabilization_strength > 0.0:
        tilt_blend = np.clip(
            profile.pelvis_stabilization_strength * pelvis_stability_weight,
            0.0,
            1.0,
        )
        base_relative_smooth = _stabilize_relative_tilt(base_relative_smooth, tilt_blend)
    else:
        tilt_blend = np.zeros(frame_count, dtype=np.float64)

    if (
        profile.pelvis_orientation_reference_mode == "source_absolute"
        and profile.pelvis_stabilization_strength <= 0.0
    ):
        # Keep the established public helper as the K1 target constructor.
        # Besides avoiding duplicate policy, this lets focused tests inject a
        # full-clip-smoothed frame-zero target into a one-frame IK run.
        target_reference = source_reference
        if profile.orientation_smoothing_mode == "rotvec_legacy":
            base_targets = build_base_orientation_targets(
                motion,
                smooth_time=profile.pelvis_orientation_smooth_time,
            )
        else:
            base_targets = build_base_orientation_targets(
                motion,
                smooth_time=profile.pelvis_orientation_smooth_time,
                smoothing_mode=profile.orientation_smoothing_mode,
            )
        base_relative_smooth = np.matmul(source_reference.T[None, :, :], base_targets)
    elif profile.pelvis_orientation_reference_mode == "robot_neutral_delta":
        target_reference = rotation(geometry.body_transforms["base"])
        base_targets = np.asarray(
            np.matmul(target_reference[None, :, :], base_relative_smooth),
            dtype=np.float64,
        )
    else:
        target_reference = source_reference
        base_targets = np.asarray(
            np.matmul(target_reference[None, :, :], base_relative_smooth),
            dtype=np.float64,
        )
    base_effective_targets = base_targets.copy()

    primary_pelvis_orientation = (
        profile.pelvis_orientation_solve_stage == "primary"
        and profile.pelvis_primary_orientation_weight > 0.0
    )
    if primary_pelvis_orientation:
        orientation_weights = (
            float(profile.pelvis_primary_orientation_weight)
            + float(profile.pelvis_stabilization_orientation_weight) * pelvis_stability_weight
        )
        if profile.pelvis_primary_dynamic_orientation_weight is not None:
            orientation_weights += (
                float(profile.pelvis_primary_dynamic_orientation_weight)
                - float(profile.pelvis_primary_orientation_weight)
            ) * (1.0 - pelvis_low_motion_weight)
    else:
        orientation_weights = (
            float(profile.pelvis_orientation_weight)
            + float(profile.pelvis_stabilization_orientation_weight) * pelvis_stability_weight
        )
    pelvis_fast_motion_weight = np.zeros(frame_count, dtype=np.float64)
    trunk_gate = (
        np.ones(frame_count, dtype=np.float64)
        if profile.trunk_position_gate == "always"
        else pelvis_stability_weight
    )
    trunk_position_blend = (
        np.clip(profile.trunk_position_strength * trunk_gate, 0.0, 1.0)
        if profile.trunk_position_mode != "source_world"
        else np.zeros(frame_count, dtype=np.float64)
    )

    torso_body_name = profile.joi_bodies[profile.torso_orientation_joi_key]
    torso_targets = _build_torso_orientation_targets(
        profile=profile,
        source=source,
        source_base_rotations=source_base_rotations,
        target_base_rotations=base_targets,
        geometry=geometry,
        robot_torso_transform=model.get_body_transform(torso_body_name),
        dt=dt,
    )
    ankle_targets = _build_ankle_orientation_targets(
        profile=profile,
        joi=joi,
        geometry=geometry,
        dt=dt,
        left_contact_confidence=left_contact_confidence,
        right_contact_confidence=right_contact_confidence,
    )

    left_hand_offset = (
        None
        if profile.left_hand_local_offset is None
        else np.asarray(profile.left_hand_local_offset, dtype=np.float64)
    )
    right_hand_offset = (
        None
        if profile.right_hand_local_offset is None
        else np.asarray(profile.right_hand_local_offset, dtype=np.float64)
    )
    pelvis_heading_reference: NDArray[np.float64] | None
    if profile.pelvis_orientation_reference_mode == "robot_neutral_delta":
        pelvis_heading_reference = None
    else:
        pelvis_heading_reference, _ = _heading_rotation_and_tilt(target_reference)

    qpos = np.empty((frame_count, model.model.nq), dtype=np.float32)
    for tick, source_frame in enumerate(source):
        if tick == 0 and profile.dmr_initial_nullspace_gain > 0.0:
            body_solver.configure_nullspace(
                home=model.get_qpos(model.rev_pri_joint_names),
                gain=profile.dmr_initial_nullspace_gain,
            )
        elif tick > 0 and profile.dmr_temporal_nullspace_gain > 0.0:
            body_solver.configure_nullspace(
                home=model.get_qpos(model.rev_pri_joint_names),
                gain=profile.dmr_temporal_nullspace_gain,
            )

        if primary_pelvis_orientation and pelvis_heading_reference is not None:
            source_relative_heading, _ = _heading_rotation_and_tilt(base_relative_smooth[tick])
            target_heading = np.matmul(pelvis_heading_reference, source_relative_heading)
            _, target_tilt = _heading_rotation_and_tilt(base_targets[tick])
            base_effective_targets[tick] = np.matmul(target_heading, target_tilt)

        targets = _body_targets(
            source_frame,
            geometry,
            profile=profile,
            effective_base_rotation=base_effective_targets[tick],
            trunk_blend=float(trunk_position_blend[tick]),
            source_base_rotation=source_base_rotations[tick],
            source_spine_reference_local=source_spine_reference_local,
            source_neck_reference_local=source_neck_reference_local,
            robot_spine_local=robot_spine_local,
            robot_neck_local=robot_neck_local,
        )
        target_positions = dict(targets)

        def populate_primary_targets(
            current_targets: tuple[tuple[str, NDArray[np.float64]], ...],
            current_positions: Mapping[str, NDArray[np.float64]],
            current_tick: int,
        ) -> None:
            body_solver.reset_targets(sync_from=model)
            _add_position_targets(
                body_solver,
                model,
                profile,
                current_targets,
                semantic_anchors,
            )
            if ankle_targets is not None and profile.ankle_orientation_stage == "primary":
                _add_primary_ankle_targets(
                    body_solver,
                    model,
                    profile,
                    current_positions,
                    {
                        "left": ankle_targets["left"][current_tick],
                        "right": ankle_targets["right"][current_tick],
                    },
                )

        populate_primary_targets(targets, target_positions, tick)
        passes = profile.initial_warmup_passes if tick == 0 else 1
        body_result = None
        if primary_pelvis_orientation and pelvis_heading_reference is None:
            for _ in range(max(1, passes)):
                body_result = body_solver.solve(
                    source_model=model,
                    joints=groups.body,
                    joint_limits=True,
                    nullspace=body_solver.nullspace_gain > 0.0,
                    base_control=True,
                )
                model.forward(body_result.qpos)

            current_heading, _ = _heading_rotation_and_tilt(
                rotation(model.get_body_transform(profile.joi_bodies["base"]))
            )
            source_relative_heading, _ = _heading_rotation_and_tilt(base_relative_smooth[tick])
            pelvis_heading_reference = np.matmul(current_heading, source_relative_heading.T)
            target_heading = np.matmul(pelvis_heading_reference, source_relative_heading)
            _, target_tilt = _heading_rotation_and_tilt(base_targets[tick])
            base_effective_targets[tick] = np.matmul(target_heading, target_tilt)
            targets = _body_targets(
                source_frame,
                geometry,
                profile=profile,
                effective_base_rotation=base_effective_targets[tick],
                trunk_blend=float(trunk_position_blend[tick]),
                source_base_rotation=source_base_rotations[tick],
                source_spine_reference_local=source_spine_reference_local,
                source_neck_reference_local=source_neck_reference_local,
                robot_spine_local=robot_spine_local,
                robot_neck_local=robot_neck_local,
            )
            target_positions = dict(targets)
            populate_primary_targets(targets, target_positions, tick)
            remaining_passes = 1
        else:
            remaining_passes = max(1, passes)

        if primary_pelvis_orientation:
            base_body_name = profile.joi_bodies["base"]
            body_solver.add_transform_target(
                base_body_name,
                _semantic_joi_transform(
                    model,
                    profile,
                    "base",
                    semantic_anchors,
                ),
                transform(target_positions["base"], base_effective_targets[tick]),
                weight=float(orientation_weights[tick]),
                axis_length=profile.pelvis_orientation_axis_length,
            )

        for _ in range(remaining_passes):
            body_result = body_solver.solve(
                source_model=model,
                joints=groups.body,
                joint_limits=True,
                nullspace=body_solver.nullspace_gain > 0.0,
                base_control=True,
            )
            model.forward(body_result.qpos)

        pelvis_orientation_weight = float(orientation_weights[tick])
        if (
            profile.pelvis_orientation_solve_stage == "legacy_post"
            and pelvis_orientation_weight > 0.0
        ):
            base_body_name = profile.joi_bodies["base"]
            body_solver.add_transform_target(
                base_body_name,
                _semantic_joi_transform(
                    model,
                    profile,
                    "base",
                    semantic_anchors,
                ),
                transform(target_positions["base"], base_targets[tick]),
                weight=pelvis_orientation_weight,
                axis_length=profile.pelvis_orientation_axis_length,
            )
            primary_step = body_solver.revolute_step
            try:
                body_solver.revolute_step = 0.0
                body_result = body_solver.solve(
                    source_model=model,
                    joints=groups.body,
                    joint_limits=False,
                    nullspace=False,
                    base_control=True,
                )
                model.forward(body_result.qpos)
            finally:
                body_solver.revolute_step = primary_step

        if torso_targets is not None and profile.torso_orientation_stage == "post" and groups.waist:
            if torso_solver is None:
                raise RuntimeError("Active torso post stage has no configured solver.")
            torso_solver.reset_targets(sync_from=model)
            torso_current = model.get_body_transform(torso_body_name)
            base_after_primary = model.get_body_transform(profile.joi_bodies["base"])
            torso_position = position(torso_current)
            torso_rotation = rotation(torso_current)
            torso_relative_target = np.matmul(base_targets[tick].T, torso_targets[tick])
            torso_post_target = np.matmul(rotation(base_after_primary), torso_relative_target)
            for axis in profile.torso_orientation_axes:
                torso_solver.add_target(
                    torso_body_name,
                    torso_position
                    + profile.torso_orientation_axis_length * torso_rotation[:, axis],
                    torso_position
                    + profile.torso_orientation_axis_length * torso_post_target[:, axis],
                    weight=profile.torso_orientation_weight,
                )
            torso_result = torso_solver.solve(
                source_model=model,
                joints=groups.waist,
                joint_limits=False,
                nullspace=False,
                base_control=False,
            )
            model.forward(torso_result.qpos, clip_position=True)

        if ankle_targets is not None and profile.ankle_orientation_stage == "post" and groups.ankle:
            if ankle_solver is None:
                raise RuntimeError("Active ankle post stage has no configured solver.")
            ankle_solver.reset_targets(sync_from=model)
            ankle_keys = {
                "left": profile.left_ankle_orientation_joi_key or "lf",
                "right": profile.right_ankle_orientation_joi_key or "rf",
            }
            for side in ("right", "left"):
                key = ankle_keys[side]
                body_name = profile.joi_bodies[key]
                current = model.get_body_transform(body_name)
                current_position = position(current)
                current_rotation = rotation(current)
                for axis in profile.ankle_orientation_axes:
                    ankle_solver.add_target(
                        body_name,
                        current_position
                        + profile.ankle_orientation_axis_length * current_rotation[:, axis],
                        current_position
                        + profile.ankle_orientation_axis_length
                        * ankle_targets[side][tick, :, axis],
                    )
            ankle_result = ankle_solver.solve(
                source_model=model,
                joints=groups.ankle,
                joint_limits=True,
                nullspace=False,
                base_control=False,
            )
            model.forward(ankle_result.qpos)

        left_source_rotation = rotation(source_frame["lw"])
        right_source_rotation = rotation(source_frame["rw"])
        left_hand_transform = model.get_body_transform(profile.joi_bodies["lh"])
        right_hand_transform = model.get_body_transform(profile.joi_bodies["rh"])
        if left_hand_offset is None:
            left_hand_offset = np.matmul(left_source_rotation.T, rotation(left_hand_transform))
        if right_hand_offset is None:
            right_hand_offset = np.matmul(right_source_rotation.T, rotation(right_hand_transform))
        left_hand_target = np.matmul(left_source_rotation, left_hand_offset)
        right_hand_target = np.matmul(right_source_rotation, right_hand_offset)

        hand_solver.reset_targets(sync_from=model)
        for body_key, current, target_rotation, anchor_local, signs in (
            (
                "lh",
                left_hand_transform,
                left_hand_target,
                profile.left_hand_anchor_local,
                profile.left_hand_axis_signs,
            ),
            (
                "rh",
                right_hand_transform,
                right_hand_target,
                profile.right_hand_anchor_local,
                profile.right_hand_axis_signs,
            ),
        ):
            current_rotation = rotation(current)
            anchor = position(current) + current_rotation @ np.asarray(
                anchor_local, dtype=np.float64
            )
            for axis, sign in enumerate(signs):
                signed_length = float(sign) * profile.hand_orientation_axis_length
                hand_solver.add_target(
                    profile.joi_bodies[body_key],
                    anchor + signed_length * current_rotation[:, axis],
                    anchor + signed_length * target_rotation[:, axis],
                )
        if profile.hand_orientation_enabled and groups.wrist:
            hand_result = hand_solver.solve(
                source_model=model,
                joints=groups.wrist,
                joint_limits=True,
                nullspace=False,
                base_control=False,
            )
            model.forward(hand_result.qpos)

        qpos[tick] = model.get_qpos().astype(np.float32, copy=False)
        if progress is not None and body_result is not None:
            progress(tick + 1, frame_count, float(body_result.error))

    smoothing_names = tuple(
        name
        for name in model.rev_pri_joint_names
        if _matches_any_token(name, profile.pelvis_stabilization_joint_smooth_tokens)
    )
    smoothing_weight = (
        np.ones(frame_count, dtype=np.float64)
        if profile.pelvis_stabilization_joint_smooth_gate == "always"
        else pelvis_stability_weight
    )
    smoothing_enabled = (
        bool(smoothing_names)
        and frame_count > 1
        and (
            profile.pelvis_stabilization_joint_median_window > 1
            or profile.pelvis_stabilization_joint_smooth_time > 0.0
        )
    )
    if smoothing_enabled:
        smoothing_indices = model.get_qpos_indices(smoothing_names)
        joint_raw = np.asarray(qpos[:, smoothing_indices], dtype=np.float64)
        joint_filtered = joint_raw.copy()
        median_window = int(profile.pelvis_stabilization_joint_median_window)
        if median_window > 1:
            if median_window % 2 == 0:
                median_window += 1
            joint_filtered = median_filter(joint_filtered, size=(median_window, 1), mode="nearest")
        if profile.pelvis_stabilization_joint_smooth_time > 0.0:
            joint_sigma = profile.pelvis_stabilization_joint_smooth_time / max(float(dt), 1e-12)
            joint_filtered = gaussian_filter1d(
                joint_filtered, sigma=joint_sigma, axis=0, mode="nearest"
            )
        joint_delta = joint_filtered - joint_raw
        max_delta = float(profile.pelvis_stabilization_joint_smooth_max_delta)
        if max_delta > 0.0:
            joint_delta = np.clip(joint_delta, -max_delta, max_delta)
        qpos[:, smoothing_indices] = (joint_raw + smoothing_weight[:, None] * joint_delta).astype(
            qpos.dtype
        )

    zeros = np.zeros(frame_count, dtype=np.float64)
    return DmrResult(
        robot_id=profile.robot_id,
        fps=motion.fps,
        seconds=motion.seconds,
        qpos=qpos,
        base_target_rotations=base_targets,
        base_effective_target_rotations=base_effective_targets,
        torso_target_rotations=torso_targets,
        pelvis_stabilization_weight=pelvis_stability_weight,
        pelvis_low_motion_weight=pelvis_low_motion_weight,
        pelvis_fast_motion_weight=pelvis_fast_motion_weight,
        pelvis_tilt_blend=tilt_blend,
        pelvis_upright_blend=zeros,
        pelvis_upright_pose_weight=zeros,
        pelvis_orientation_weight=np.asarray(orientation_weights, dtype=np.float64),
        pelvis_decoupled_solve_blend=zeros,
        trunk_position_blend=np.asarray(trunk_position_blend, dtype=np.float64),
        source_provider=source_provider,
        left_ankle_target_rotations=(None if ankle_targets is None else ankle_targets["left"]),
        right_ankle_target_rotations=(None if ankle_targets is None else ankle_targets["right"]),
    )


__all__ = ["DmrProgress", "DmrResult", "build_base_orientation_targets", "run_dmr"]
