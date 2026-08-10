"""Final arm-only self-collision refinement after FPA and grounding.

The supported robot profiles keep the optional post-ground micro-lift disabled.
Stage 8 applies the same signed-distance arm solve as Stage 3 without changing
the floating base or lower body.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.native import BackendPreference, BackendSelection
from core_retarget.robots.registry import get_robot
from core_retarget.stages.initial_collision import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
)
from core_retarget.stages.initial_collision import (
    run_initial_collision as run_initial_collision_stage,
)

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FinalCollisionProgress = Callable[[int, int, int, float], None]

_REFERENCE_MUJOCO_VERSION = "3.6.0"


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_bool(value: ArrayLike) -> BoolArray:
    result = np.array(value, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _contact_segments(label: BoolArray) -> IntArray:
    values = np.asarray(label, dtype=np.bool_).reshape(-1)
    padded = np.concatenate([np.asarray([False]), values, np.asarray([False])]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return _readonly_int(np.column_stack([starts, ends]).reshape(-1, 2))


def _coerce_binary(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.hasobject or raw.dtype.kind in "SUc":
        raise MotionValidationError(f"{name} must contain boolean or binary numeric values.")
    if raw.dtype.kind in "fiu":
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any((numeric != 0.0) & (numeric != 1.0)):
            raise MotionValidationError(f"{name} must contain only zero or one.")
    return _readonly_bool(raw)


def _canonical_contacts(
    frame_count: int,
    labels: ArrayLike | None,
    confidence: ArrayLike | None,
    flight: ArrayLike | None,
) -> tuple[BoolArray | None, FloatArray | None, BoolArray | None]:
    if labels is None and confidence is None and flight is None:
        return None, None, None
    if labels is None or confidence is None:
        raise MotionValidationError(
            "Final-collision contact labels and confidence must be provided together."
        )

    label_values = _coerce_binary(labels, name="Final-collision contact labels")
    confidence_values = _readonly_float(confidence)
    if label_values.ndim != 2 or label_values.shape[0] != frame_count:
        raise MotionValidationError(
            "Final-collision contact labels must have shape (frames, 2) or (frames, 4)."
        )
    if label_values.shape[1] not in (2, 4):
        raise MotionValidationError(
            "Final-collision contact labels must have shape (frames, 2) or (frames, 4)."
        )
    if confidence_values.shape != label_values.shape:
        raise MotionValidationError(
            "Final-collision contact confidence must match the contact-label shape."
        )
    if not np.isfinite(confidence_values).all() or np.any(
        (confidence_values < 0.0) | (confidence_values > 1.0)
    ):
        raise MotionValidationError(
            "Final-collision contact confidence must be finite and lie in [0, 1]."
        )

    if label_values.shape[1] == 2:
        canonical_labels = np.zeros((frame_count, 4), dtype=np.bool_)
        canonical_confidence = np.zeros((frame_count, 4), dtype=np.float64)
        canonical_labels[:, :2] = label_values
        canonical_confidence[:, :2] = confidence_values
        label_values = _readonly_bool(canonical_labels)
        confidence_values = _readonly_float(canonical_confidence)

    flight_values = (
        _readonly_bool(~(label_values[:, 0] | label_values[:, 1]))
        if flight is None
        else _coerce_binary(flight, name="Final-collision flight labels").reshape(-1)
    )
    if flight_values.shape != (frame_count,):
        raise MotionValidationError(
            f"Final-collision flight labels must have shape ({frame_count},)."
        )
    return label_values, confidence_values, flight_values


@dataclass(frozen=True, slots=True)
class FinalCollisionDiagnostics:
    """Stage 8 collision facts, separate from the Stage 3 public type."""

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

    @classmethod
    def from_initial(
        cls,
        diagnostics: InitialCollisionDiagnostics,
    ) -> FinalCollisionDiagnostics:
        """Copy shared-engine diagnostics into the Stage 8 contract."""

        return cls(
            backend=diagnostics.backend,
            distance_backend=diagnostics.distance_backend,
            ik_backend=diagnostics.ik_backend,
            trajectory_backend=diagnostics.trajectory_backend,
            root_geom_count=diagnostics.root_geom_count,
            collision_geom_count=diagnostics.collision_geom_count,
            raw_candidate_pair_count=diagnostics.raw_candidate_pair_count,
            candidate_pair_count=diagnostics.candidate_pair_count,
            arm_joint_names=tuple(diagnostics.arm_joint_names),
            input_violations=diagnostics.input_violations,
            input_max_frame_violations=diagnostics.input_max_frame_violations,
            output_violations=diagnostics.output_violations,
            output_max_frame_violations=diagnostics.output_max_frame_violations,
            passes=tuple(diagnostics.passes),
        )


@dataclass(frozen=True, slots=True)
class FinalCollisionResult:
    """Final Stage 8 trajectory and optional source-derived contact state."""

    robot_id: str
    fps: float
    seconds: FloatArray
    qpos: FloatArray
    diagnostics: FinalCollisionDiagnostics
    contact_labels: BoolArray | None = None
    contact_confidence: FloatArray | None = None
    flight_labels: BoolArray | None = None

    def __post_init__(self) -> None:
        robot = get_robot(self.robot_id)
        seconds = _readonly_float(self.seconds).reshape(-1)
        qpos = _readonly_float(self.qpos)
        if qpos.shape != (len(seconds), robot.expected_nq):
            raise ValueError(
                f"final-collision qpos must have shape ({len(seconds)}, {robot.expected_nq})"
            )
        if len(seconds) == 0:
            raise ValueError("final-collision result must contain at least one frame")
        if not np.isfinite(seconds).all() or not np.isfinite(qpos).all():
            raise ValueError("final-collision arrays must contain only finite values")
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0.0):
            raise ValueError("final-collision seconds must be strictly increasing")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("final-collision fps must be finite and positive")

        try:
            labels, confidence, flight = _canonical_contacts(
                len(seconds),
                self.contact_labels,
                self.contact_confidence,
                self.flight_labels,
            )
        except MotionValidationError as error:
            raise ValueError(str(error)) from error

        object.__setattr__(self, "robot_id", robot.robot_id)
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "contact_labels", labels)
        object.__setattr__(self, "contact_confidence", confidence)
        object.__setattr__(self, "flight_labels", flight)

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 8 archive arrays when contacts are known."""

        arrays: dict[str, NDArray[np.generic]] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_cc_fpa_array": self.qpos,
        }
        if self.contact_labels is None:
            return arrays

        assert self.contact_confidence is not None
        assert self.flight_labels is not None
        left_label = self.contact_labels[:, 0]
        right_label = self.contact_labels[:, 1]
        left_confidence = self.contact_confidence[:, 0]
        right_confidence = self.contact_confidence[:, 1]
        arrays.update(
            {
                "r_contact_label_merged": right_label,
                "l_contact_label_merged": left_label,
                "r_contact_confidence": right_confidence,
                "l_contact_confidence": left_confidence,
                "flight_contact_label": self.flight_labels,
                "right_contact_confidence": right_confidence,
                "right_contact_label": right_label,
                "right_contact_segments": _contact_segments(right_label),
                "left_contact_confidence": left_confidence,
                "left_contact_label": left_label,
                "left_contact_segments": _contact_segments(left_label),
            }
        )
        return arrays


def _validate_stage_inputs(
    qpos: ArrayLike,
    seconds: ArrayLike,
    *,
    robot_id: str,
    fps: float,
) -> tuple[str, FloatArray, FloatArray, float]:
    robot = get_robot(robot_id)
    trajectory = np.asarray(qpos, dtype=np.float64)
    time_values = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if trajectory.shape != (len(time_values), robot.expected_nq):
        raise MotionValidationError(
            "Stage 8 qpos must have shape "
            f"(frames, {robot.expected_nq}); found {trajectory.shape}."
        )
    if len(time_values) == 0:
        raise MotionValidationError("Stage 8 input must contain at least one frame.")
    if not np.isfinite(trajectory).all() or not np.isfinite(time_values).all():
        raise MotionValidationError("Stage 8 input contains NaN or infinity.")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise MotionValidationError("Stage 8 seconds must be strictly increasing.")
    try:
        fps_value = float(fps)
    except (TypeError, ValueError, OverflowError) as error:
        raise MotionValidationError("Stage 8 fps must be finite and positive.") from error
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise MotionValidationError("Stage 8 fps must be finite and positive.")
    return (
        robot.robot_id,
        trajectory.copy(order="C"),
        time_values.copy(order="C"),
        fps_value,
    )


def _validate_reference_environment() -> None:
    version = str(getattr(mujoco, "__version__", "unknown"))
    if version != _REFERENCE_MUJOCO_VERSION:
        raise ConfigurationError(
            "Final collision refinement is numerically qualified with MuJoCo "
            f"{_REFERENCE_MUJOCO_VERSION}; the loaded version is {version}."
        )


def run_final_collision(
    qpos: ArrayLike,
    seconds: ArrayLike,
    *,
    robot_id: str,
    fps: float,
    contact_labels: ArrayLike | None = None,
    contact_confidence: ArrayLike | None = None,
    flight_labels: ArrayLike | None = None,
    progress: FinalCollisionProgress | None = None,
    backend: BackendPreference | BackendSelection = "python",
) -> FinalCollisionResult:
    """Run arm refinement on a Stage 7 FPA trajectory.

    Contact arrays are carried through for the Stage 8 archive and
    production export.  They never influence collision detection or IK.
    """

    _validate_reference_environment()
    normalized_robot, trajectory, time_values, fps_value = _validate_stage_inputs(
        qpos,
        seconds,
        robot_id=robot_id,
        fps=fps,
    )
    labels, confidence, flight = _canonical_contacts(
        len(time_values),
        contact_labels,
        contact_confidence,
        flight_labels,
    )
    shared_result = run_initial_collision_stage(
        trajectory,
        time_values,
        robot_id=normalized_robot,
        fps=fps_value,
        progress=progress,
        backend=backend,
    )
    return FinalCollisionResult(
        robot_id=normalized_robot,
        fps=fps_value,
        seconds=shared_result.seconds,
        qpos=shared_result.qpos,
        diagnostics=FinalCollisionDiagnostics.from_initial(shared_result.diagnostics),
        contact_labels=labels,
        contact_confidence=confidence,
        flight_labels=flight,
    )


__all__ = [
    "FinalCollisionDiagnostics",
    "FinalCollisionProgress",
    "FinalCollisionResult",
    "run_final_collision",
]
