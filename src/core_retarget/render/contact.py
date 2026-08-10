"""Validated contact-state data used by the review renderer.

The overlay is driven by the source SOMA contact schedule, never by MuJoCo's
instantaneous collision contacts. This distinction matters when reviewing
intermediate trajectories before grounding and FPA have run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

from core_retarget.exceptions import MotionValidationError
from core_retarget.motion.contacts import build_contact_schedule
from core_retarget.motion.soma import SomaMotion
from core_retarget.mujoco.model import MujocoModel
from core_retarget.robots.registry import get_robot

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CONTACT_CHANNEL_NAMES = ("left_foot", "right_foot", "left_hand", "right_hand")
CONTACT_SHORT_NAMES = ("LF", "RF", "LH", "RH")
CONTACT_COLORS_RGB = (
    (54, 153, 255),
    (255, 91, 82),
    (80, 220, 255),
    (255, 165, 95),
)

_LOWER_BODY_JOINT_TOKENS = ("hip", "knee", "ankle", "toe")
_MACRO_SEGMENT_CONFIG = {
    "root_joint_smooth_time": 0.25,
    "contact_smooth_time": 0.35,
    "min_segment_time": 0.40,
    "max_segments": 8,
    "split_penalty_ratio": 0.08,
}


def _readonly_bool(value: ArrayLike) -> BoolArray:
    result = np.array(value, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PreviewContactState:
    """Source-derived LF/RF/LH/RH state and review-only macro segments."""

    fps: float
    seconds: FloatArray
    labels: BoolArray
    confidence: FloatArray
    availability: BoolArray
    flight: BoolArray
    segment_ranges: IntArray
    segment_boundaries: FloatArray
    contact_source: str
    hand_contact_source: str

    def __post_init__(self) -> None:
        try:
            fps = float(self.fps)
        except (TypeError, ValueError, OverflowError) as error:
            raise MotionValidationError(
                "Preview contact FPS must be finite and positive"
            ) from error
        if not np.isfinite(fps) or fps <= 0.0:
            raise MotionValidationError("Preview contact FPS must be finite and positive")

        seconds = _readonly_float(self.seconds).reshape(-1)
        labels = _readonly_bool(self.labels)
        confidence = _readonly_float(self.confidence)
        availability = _readonly_bool(self.availability).reshape(-1)
        flight = _readonly_bool(self.flight).reshape(-1)
        ranges = _readonly_int(self.segment_ranges)
        boundaries = _readonly_float(self.segment_boundaries).reshape(-1)
        frame_count = len(seconds)

        if labels.shape != (frame_count, 4):
            raise MotionValidationError(
                f"Preview contact labels must have shape ({frame_count}, 4)"
            )
        if confidence.shape != (frame_count, 4):
            raise MotionValidationError(
                f"Preview contact confidence must have shape ({frame_count}, 4)"
            )
        if availability.shape != (4,):
            raise MotionValidationError("Preview contact availability must have shape (4,)")
        if flight.shape != (frame_count,):
            raise MotionValidationError(f"Preview flight labels must have shape ({frame_count},)")
        if not np.isfinite(seconds).all() or not np.isfinite(confidence).all():
            raise MotionValidationError("Preview contact arrays must contain only finite values")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise MotionValidationError("Preview contact confidence must lie in [0, 1]")
        if np.any(labels[:, ~availability]) or np.any(confidence[:, ~availability] != 0.0):
            raise MotionValidationError(
                "Unavailable preview contact channels must be false with zero confidence"
            )
        if ranges.ndim != 2 or ranges.shape[1:] != (2,):
            raise MotionValidationError("Preview contact segment ranges must have shape (N, 2)")
        if len(ranges) == 0 or boundaries.shape != (len(ranges) + 1,):
            raise MotionValidationError(
                "Preview contact segment boundaries must contain one more item than ranges"
            )
        if (
            int(ranges[0, 0]) != 0
            or int(ranges[-1, 1]) != frame_count
            or np.any(ranges[:, 0] >= ranges[:, 1])
            or np.any(ranges[1:, 0] != ranges[:-1, 1])
        ):
            raise MotionValidationError(
                "Preview contact segment ranges must partition all frames contiguously"
            )
        if (
            not np.isfinite(boundaries).all()
            or float(boundaries[0]) != 0.0
            or float(boundaries[-1]) != 1.0
            or np.any(np.diff(boundaries) <= 0.0)
        ):
            raise MotionValidationError(
                "Preview contact segment boundaries must increase from 0 to 1"
            )
        if not self.contact_source:
            raise MotionValidationError("Preview contact source must not be empty")
        if not self.hand_contact_source:
            raise MotionValidationError("Preview hand-contact source must not be empty")

        seconds.setflags(write=False)
        labels.setflags(write=False)
        confidence.setflags(write=False)
        availability.setflags(write=False)
        flight.setflags(write=False)
        ranges.setflags(write=False)
        boundaries.setflags(write=False)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "flight", flight)
        object.__setattr__(self, "segment_ranges", ranges)
        object.__setattr__(self, "segment_boundaries", boundaries)

    @property
    def frame_count(self) -> int:
        """Number of frames covered by the contact schedule."""

        return int(len(self.seconds))

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return an object-free contact artifact suitable for independent rendering."""

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "contact_label_names": np.asarray(CONTACT_CHANNEL_NAMES),
            "contact_labels": self.labels,
            "contact_confidence": self.confidence,
            "contact_availability": self.availability,
            "flight_labels": self.flight,
            "contact_segment_frames": self.segment_ranges,
            "contact_segments": self.segment_boundaries,
            "contact_source": np.asarray(self.contact_source),
            "hand_contact_source": np.asarray(self.hand_contact_source),
        }


def _raw_single_frame_contacts(motion: SomaMotion) -> tuple[BoolArray, BoolArray, str, bool]:
    left = np.zeros(motion.frame_count, dtype=np.bool_)
    right = np.zeros(motion.frame_count, dtype=np.bool_)
    contacts = motion.foot_contacts
    if contacts is None:
        return left, right, "unavailable_single_frame", False
    values = np.asarray(contacts, dtype=bool)
    if values.shape[1] >= 6:
        left[:] = np.any(values[:, 1:3], axis=1)
        right[:] = np.any(values[:, 4:6], axis=1)
        return left, right, "kimodo_toe_contacts_6ch", True
    if values.shape[1] >= 4:
        left[:] = values[:, 1]
        right[:] = values[:, 3]
        return left, right, "kimodo_toe_contacts_4ch", True
    return left, right, "unavailable_single_frame", False


def _lower_body_qpos_indices(robot_id: str) -> NDArray[np.int32]:
    model = MujocoModel.from_robot(robot_id)
    names = tuple(
        name
        for name in model.rev_pri_joint_names
        if any(token in name.lower() for token in _LOWER_BODY_JOINT_TOKENS)
    )
    if not names:
        raise MotionValidationError(f"Robot {robot_id!r} has no CATO lower-body joints")
    return model.get_qpos_indices(names)


def _macro_segments(
    qpos: FloatArray,
    labels: BoolArray,
    fps: float,
    lower_body_indices: NDArray[np.int32],
) -> tuple[IntArray, FloatArray]:
    """Apply penalized piecewise-constant CATO segmentation."""

    frame_count = len(qpos)
    if frame_count == 1:
        return (
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        )

    dt = 1.0 / fps
    root_velocity = np.gradient(qpos[:, 0:3], dt, axis=0)
    lower_body = np.unwrap(qpos[:, lower_body_indices], axis=0)
    lower_body_velocity = np.gradient(lower_body, dt, axis=0)

    def smooth(values: FloatArray, seconds: float) -> FloatArray:
        sigma = max(float(seconds) * fps, 1e-6)
        return np.asarray(
            gaussian_filter1d(values, sigma=sigma, axis=0, mode="nearest", truncate=3.0),
            dtype=np.float64,
        )

    root_xy_speed = smooth(
        np.linalg.norm(root_velocity[:, :2], axis=1),
        _MACRO_SEGMENT_CONFIG["root_joint_smooth_time"],
    )
    root_abs_z_speed = smooth(
        np.abs(root_velocity[:, 2]),
        _MACRO_SEGMENT_CONFIG["root_joint_smooth_time"],
    )
    lower_body_speed_rms = smooth(
        np.sqrt(np.mean(lower_body_velocity * lower_body_velocity, axis=1)),
        _MACRO_SEGMENT_CONFIG["root_joint_smooth_time"],
    )
    transition = np.zeros(frame_count, dtype=np.float64)
    transition[1:] = np.any(labels[1:] != labels[:-1], axis=1)
    transition_rate = smooth(
        transition * fps,
        _MACRO_SEGMENT_CONFIG["contact_smooth_time"],
    )
    flight_density = smooth(
        (~(labels[:, 0] | labels[:, 1])).astype(np.float64),
        _MACRO_SEGMENT_CONFIG["contact_smooth_time"],
    )

    raw_features = (
        root_xy_speed,
        root_abs_z_speed,
        lower_body_speed_rms,
        transition_rate,
        flight_density,
    )
    feature_weights = (1.0, 0.35, 0.50, 0.20, 0.80)
    normalized: list[FloatArray] = []
    for feature, weight in zip(raw_features, feature_weights, strict=True):
        scale = float(np.quantile(feature, 0.90))
        values = np.zeros_like(feature) if scale <= 1e-8 else np.clip(feature / scale, 0.0, 2.0)
        normalized.append(float(weight) * values)
    features = np.stack(normalized, axis=1)

    minimum_frames = min(
        frame_count,
        max(2, int(round(_MACRO_SEGMENT_CONFIG["min_segment_time"] * fps))),
    )
    maximum_count = max(
        1,
        min(int(_MACRO_SEGMENT_CONFIG["max_segments"]), frame_count // minimum_frames),
    )
    prefix = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, axis=0)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(np.sum(features * features, axis=1))])

    def cost(start: int, end: int) -> float:
        length = end - start
        total = prefix[end] - prefix[start]
        squared = prefix_sq[end] - prefix_sq[start]
        return max(0.0, float(squared - np.dot(total, total) / length))

    dynamic = np.full((maximum_count + 1, frame_count + 1), np.inf, dtype=np.float64)
    previous = np.full((maximum_count + 1, frame_count + 1), -1, dtype=np.int64)
    dynamic[0, 0] = 0.0
    for count in range(1, maximum_count + 1):
        for end in range(count * minimum_frames, frame_count + 1):
            first_start = (count - 1) * minimum_frames
            last_start = end - minimum_frames
            for start in range(first_start, last_start + 1):
                if not np.isfinite(dynamic[count - 1, start]):
                    continue
                candidate = dynamic[count - 1, start] + cost(start, end)
                if candidate < dynamic[count, end]:
                    dynamic[count, end] = candidate
                    previous[count, end] = start

    one_segment_cost = float(dynamic[1, frame_count])
    if one_segment_cost <= 1e-10:
        selected_count = 1
    else:
        penalty = float(_MACRO_SEGMENT_CONFIG["split_penalty_ratio"]) * one_segment_cost
        objectives = np.asarray(
            [
                dynamic[count, frame_count] + penalty * (count - 1)
                for count in range(1, maximum_count + 1)
            ],
            dtype=np.float64,
        )
        selected_count = int(np.argmin(objectives)) + 1

    frame_boundaries = [frame_count]
    frame_end = frame_count
    for count in range(selected_count, 0, -1):
        frame_end = int(previous[count, frame_end])
        frame_boundaries.append(frame_end)
    frame_boundaries.reverse()
    if frame_boundaries[0] != 0 or frame_boundaries[-1] != frame_count:
        raise MotionValidationError(f"Invalid preview contact macro boundaries: {frame_boundaries}")
    ranges = np.asarray(
        [
            [frame_boundaries[index], frame_boundaries[index + 1]]
            for index in range(len(frame_boundaries) - 1)
        ],
        dtype=np.int64,
    )
    boundaries = np.asarray(
        [round(frame / float(frame_count), 8) for frame in frame_boundaries],
        dtype=np.float64,
    )
    boundaries[0], boundaries[-1] = 0.0, 1.0
    return ranges, boundaries


def build_preview_contact_state(
    motion: SomaMotion,
    *,
    qpos: ArrayLike,
    robot_id: str,
) -> PreviewContactState:
    """Build the exact source-derived state consumed by review visualization."""

    trajectory = np.asarray(qpos, dtype=np.float64)
    frame_count = motion.frame_count
    expected_nq = get_robot(robot_id).expected_nq
    if trajectory.shape != (frame_count, expected_nq):
        raise MotionValidationError(
            f"Preview qpos must have shape ({frame_count}, {expected_nq}); found {trajectory.shape}"
        )
    if not np.isfinite(trajectory).all():
        raise MotionValidationError("Preview qpos contains NaN or infinity")

    if frame_count >= 2:
        schedule = build_contact_schedule(motion)
        left = schedule.left_contact_label
        right = schedule.right_contact_label
        left_confidence = schedule.left_confidence
        right_confidence = schedule.right_confidence
        flight = schedule.flight_label if schedule.flight_label is not None else ~(left | right)
        contact_source = schedule.contact_source
        foot_available = True
    else:
        left, right, contact_source, foot_available = _raw_single_frame_contacts(motion)
        left_confidence = left.astype(np.float64)
        right_confidence = right.astype(np.float64)
        flight = ~(left | right)

    labels = np.column_stack(
        [left, right, np.zeros(frame_count, dtype=bool), np.zeros(frame_count, dtype=bool)]
    )
    confidence = np.column_stack(
        [
            left_confidence,
            right_confidence,
            np.zeros(frame_count, dtype=np.float64),
            np.zeros(frame_count, dtype=np.float64),
        ]
    )
    availability = np.asarray(
        [foot_available, foot_available, False, False],
        dtype=np.bool_,
    )
    lower_body_indices = _lower_body_qpos_indices(robot_id)
    ranges, boundaries = _macro_segments(
        trajectory,
        np.asarray(labels, dtype=np.bool_),
        float(motion.fps),
        lower_body_indices,
    )
    return PreviewContactState(
        fps=motion.fps,
        seconds=motion.seconds,
        labels=labels,
        confidence=confidence,
        availability=availability,
        flight=flight,
        segment_ranges=ranges,
        segment_boundaries=boundaries,
        contact_source=contact_source,
        hand_contact_source="unavailable_soma_default_false",
    )


__all__ = [
    "CONTACT_CHANNEL_NAMES",
    "CONTACT_COLORS_RGB",
    "CONTACT_SHORT_NAMES",
    "PreviewContactState",
    "build_preview_contact_state",
]
