"""Robot-space target extraction after the first collision pass.

The stage uses forward kinematics to convert raw DMR and Stage 3 trajectories
into root, ankle, and toe world-position trajectories. The root landmark is
the midpoint of the two hip landmarks when ``base_between_hips=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rimkit.exceptions import MotionValidationError
from rimkit.mujoco.model import MujocoModel
from rimkit.robots.profiles import DmrProfile, get_dmr_profile

FloatArray = NDArray[np.float64]


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TargetTrajectoriesResult:
    """Immutable Stage 4 robot landmark trajectories."""

    robot_id: str
    fps: float
    seconds: FloatArray
    root: FloatArray
    right_ankle: FloatArray
    left_ankle: FloatArray
    right_toe: FloatArray
    left_toe: FloatArray
    root_smoothed: FloatArray
    right_ankle_smoothed: FloatArray
    left_ankle_smoothed: FloatArray
    right_toe_smoothed: FloatArray
    left_toe_smoothed: FloatArray

    def __post_init__(self) -> None:
        try:
            fps = float(self.fps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("target-trajectory fps must be finite and positive") from exc
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("target-trajectory fps must be finite and positive")
        object.__setattr__(self, "fps", fps)

        seconds = _readonly_float(self.seconds).reshape(-1)
        if seconds.size == 0:
            raise ValueError("target trajectories must contain at least one frame")
        if not np.isfinite(seconds).all():
            raise ValueError("target-trajectory seconds contain NaN or infinity")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("target-trajectory seconds must be strictly increasing")
        object.__setattr__(self, "seconds", seconds)

        frame_count = len(seconds)
        trajectory_fields = (
            "root",
            "right_ankle",
            "left_ankle",
            "right_toe",
            "left_toe",
            "root_smoothed",
            "right_ankle_smoothed",
            "left_ankle_smoothed",
            "right_toe_smoothed",
            "left_toe_smoothed",
        )
        for field_name in trajectory_fields:
            value = _readonly_float(getattr(self, field_name))
            if value.shape != (frame_count, 3):
                raise ValueError(
                    f"target-trajectory field {field_name} must have shape ({frame_count}, 3)"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"target-trajectory field {field_name} contains NaN or infinity")
            object.__setattr__(self, field_name, value)

    @property
    def frame_count(self) -> int:
        """Number of trajectory frames."""

        return int(self.seconds.shape[0])

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 4 archive arrays."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "p_root_trgt_array": self.root,
            "p_ra_trgt_array": self.right_ankle,
            "p_la_trgt_array": self.left_ankle,
            "p_rt_trgt_array": self.right_toe,
            "p_lt_trgt_array": self.left_toe,
            "p_root_trgt_smt_array": self.root_smoothed,
            "p_ra_trgt_smt_array": self.right_ankle_smoothed,
            "p_la_trgt_smt_array": self.left_ankle_smoothed,
            "p_rt_trgt_smt_array": self.right_toe_smoothed,
            "p_lt_trgt_smt_array": self.left_toe_smoothed,
        }


def _validate_inputs(
    qpos_dmr: ArrayLike,
    qpos_smoothed: ArrayLike,
    seconds: ArrayLike,
    fps: float,
    profile: DmrProfile,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    raw = np.asarray(qpos_dmr, dtype=np.float64)
    smoothed = np.asarray(qpos_smoothed, dtype=np.float64)
    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    expected_shape = (len(time_values), profile.qpos_dim)
    if raw.shape != expected_shape:
        raise MotionValidationError(
            f"Stage 4 DMR qpos must have shape {expected_shape}; found {raw.shape}."
        )
    if smoothed.shape != expected_shape:
        raise MotionValidationError(
            "Stage 4 collision-refined qpos must have shape "
            f"{expected_shape}; found {smoothed.shape}."
        )
    if len(time_values) == 0:
        raise MotionValidationError("Stage 4 input must contain at least one frame.")
    if not np.isfinite(raw).all() or not np.isfinite(smoothed).all():
        raise MotionValidationError("Stage 4 qpos contains NaN or infinity.")
    if not np.isfinite(time_values).all():
        raise MotionValidationError("Stage 4 seconds contain NaN or infinity.")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise MotionValidationError("Stage 4 seconds must be strictly increasing.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise MotionValidationError("Stage 4 fps must be finite and positive.")
    return (
        raw.copy(order="C"),
        smoothed.copy(order="C"),
        time_values.copy(order="C"),
    )


def _landmarks(
    model: MujocoModel,
    profile: DmrProfile,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    positions = {
        key: model.get_body_transform(profile.joi_bodies[key])[:3, 3]
        for key in ("rp", "lp", "ra", "la", "rt", "lt")
    }
    root = 0.5 * (positions["rp"] + positions["lp"])
    return (
        np.asarray(root, dtype=np.float64),
        np.asarray(positions["ra"], dtype=np.float64),
        np.asarray(positions["la"], dtype=np.float64),
        np.asarray(positions["rt"], dtype=np.float64),
        np.asarray(positions["lt"], dtype=np.float64),
    )


def run_target_trajectories(
    qpos_dmr: ArrayLike,
    qpos_smoothed: ArrayLike,
    seconds: ArrayLike,
    *,
    robot_id: str,
    fps: float,
) -> TargetTrajectoriesResult:
    """Extract raw and Stage 3 robot landmark trajectories."""

    profile = get_dmr_profile(robot_id)
    raw, smoothed, time_values = _validate_inputs(
        qpos_dmr,
        qpos_smoothed,
        seconds,
        fps,
        profile,
    )
    model = MujocoModel.from_robot(profile.robot_id)
    frame_count = len(time_values)
    raw_positions = np.empty((5, frame_count, 3), dtype=np.float64)
    smoothed_positions = np.empty((5, frame_count, 3), dtype=np.float64)

    for tick in range(frame_count):
        model.forward(raw[tick])
        for index, position in enumerate(_landmarks(model, profile)):
            raw_positions[index, tick] = position

        model.forward(smoothed[tick])
        for index, position in enumerate(_landmarks(model, profile)):
            smoothed_positions[index, tick] = position

    return TargetTrajectoriesResult(
        robot_id=profile.robot_id,
        fps=float(fps),
        seconds=time_values,
        root=raw_positions[0],
        right_ankle=raw_positions[1],
        left_ankle=raw_positions[2],
        right_toe=raw_positions[3],
        left_toe=raw_positions[4],
        root_smoothed=smoothed_positions[0],
        right_ankle_smoothed=smoothed_positions[1],
        left_ankle_smoothed=smoothed_positions[2],
        right_toe_smoothed=smoothed_positions[3],
        left_toe_smoothed=smoothed_positions[4],
    )


__all__ = ["TargetTrajectoriesResult", "run_target_trajectories"]
