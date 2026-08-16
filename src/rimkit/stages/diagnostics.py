"""Final toe-trajectory diagnostics from the research Stage 9 cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rimkit.exceptions import MotionValidationError
from rimkit.mujoco.model import MujocoModel
from rimkit.robots.profiles import get_dmr_profile

FloatArray = NDArray[np.float64]


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DiagnosticTrajectoriesResult:
    """Actual toe positions immediately before and after final collision."""

    robot_id: str
    fps: float
    seconds: FloatArray
    right_toe_before_collision: FloatArray
    left_toe_before_collision: FloatArray
    right_toe_final: FloatArray
    left_toe_final: FloatArray

    def __post_init__(self) -> None:
        fps = float(self.fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("diagnostic fps must be finite and positive")
        object.__setattr__(self, "fps", fps)
        seconds = _readonly_float(self.seconds).reshape(-1)
        if seconds.size == 0:
            raise ValueError("diagnostics must contain at least one frame")
        if not np.isfinite(seconds).all():
            raise ValueError("diagnostic seconds contain NaN or infinity")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("diagnostic seconds must be strictly increasing")
        object.__setattr__(self, "seconds", seconds)
        for field_name in (
            "right_toe_before_collision",
            "left_toe_before_collision",
            "right_toe_final",
            "left_toe_final",
        ):
            value = _readonly_float(getattr(self, field_name))
            if value.shape != (len(seconds), 3):
                raise ValueError(
                    f"diagnostic field {field_name} must have shape ({len(seconds)}, 3)"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"diagnostic field {field_name} contains NaN or infinity")
            object.__setattr__(self, field_name, value)

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 9 archive arrays."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "p_rt_fpa_actual_array": self.right_toe_before_collision,
            "p_lt_fpa_actual_array": self.left_toe_before_collision,
            "p_rt_cc_actual_array": self.right_toe_final,
            "p_lt_cc_actual_array": self.left_toe_final,
        }


def _validate_qpos(
    value: ArrayLike,
    *,
    name: str,
    frame_count: int,
    qpos_dim: int,
) -> FloatArray:
    qpos = np.asarray(value, dtype=np.float64)
    expected_shape = (frame_count, qpos_dim)
    if qpos.shape != expected_shape:
        raise MotionValidationError(
            f"Stage 9 {name} qpos must have shape {expected_shape}; found {qpos.shape}."
        )
    if not np.isfinite(qpos).all():
        raise MotionValidationError(f"Stage 9 {name} qpos contains NaN or infinity.")
    return qpos.copy(order="C")


def run_diagnostic_trajectories(
    qpos_before_collision: ArrayLike,
    qpos_final: ArrayLike,
    seconds: ArrayLike,
    *,
    robot_id: str,
    fps: float,
) -> DiagnosticTrajectoriesResult:
    """Evaluate actual left/right toe-body positions for Stage 7 and Stage 8."""

    profile = get_dmr_profile(robot_id)
    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if len(time_values) == 0:
        raise MotionValidationError("Stage 9 input must contain at least one frame.")
    if not np.isfinite(time_values).all():
        raise MotionValidationError("Stage 9 seconds contain NaN or infinity.")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise MotionValidationError("Stage 9 seconds must be strictly increasing.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise MotionValidationError("Stage 9 fps must be finite and positive.")
    before = _validate_qpos(
        qpos_before_collision,
        name="pre-collision",
        frame_count=len(time_values),
        qpos_dim=profile.qpos_dim,
    )
    final = _validate_qpos(
        qpos_final,
        name="final",
        frame_count=len(time_values),
        qpos_dim=profile.qpos_dim,
    )

    model = MujocoModel.from_robot(profile.robot_id)
    before_right = np.empty((len(time_values), 3), dtype=np.float64)
    before_left = np.empty((len(time_values), 3), dtype=np.float64)
    final_right = np.empty((len(time_values), 3), dtype=np.float64)
    final_left = np.empty((len(time_values), 3), dtype=np.float64)
    right_body = profile.joi_bodies["rt"]
    left_body = profile.joi_bodies["lt"]
    for tick in range(len(time_values)):
        model.forward(before[tick])
        before_right[tick] = model.get_body_transform(right_body)[:3, 3]
        before_left[tick] = model.get_body_transform(left_body)[:3, 3]
        model.forward(final[tick])
        final_right[tick] = model.get_body_transform(right_body)[:3, 3]
        final_left[tick] = model.get_body_transform(left_body)[:3, 3]

    return DiagnosticTrajectoriesResult(
        robot_id=profile.robot_id,
        fps=float(fps),
        seconds=time_values,
        right_toe_before_collision=before_right,
        left_toe_before_collision=before_left,
        right_toe_final=final_right,
        left_toe_final=final_left,
    )


__all__ = ["DiagnosticTrajectoriesResult", "run_diagnostic_trajectories"]
