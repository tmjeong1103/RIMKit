"""Toe-primary contact preprocessing for SOMA motion.

Kimodo toe and toe-end labels remain authoritative. The stage supplements only
frames where both source toes are off and builds confidence ramps consumed by
the contact-aware retargeting stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, median_filter  # type: ignore[import-untyped]

from core_retarget.exceptions import MotionValidationError
from core_retarget.motion.soma import SomaMotion
from core_retarget.motion.soma_joints import SomaJoiTrajectory, extract_soma_joi

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _readonly_float(array: NDArray[np.generic]) -> FloatArray:
    value = np.array(array, dtype=np.float64, copy=True, order="C")
    value.setflags(write=False)
    return value


def _readonly_bool(array: NDArray[np.generic]) -> BoolArray:
    value = np.array(array, dtype=np.bool_, copy=True, order="C")
    value.setflags(write=False)
    return value


def _readonly_int(array: NDArray[np.generic]) -> IntArray:
    value = np.array(array, dtype=np.int64, copy=True, order="C")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class ContactSchedule:
    """Immutable result of the SOMA toe-contact preprocessing stage."""

    fps: float
    seconds: FloatArray
    contact_source: str
    base_positions: FloatArray
    right_ankle_positions: FloatArray
    left_ankle_positions: FloatArray
    right_toe_positions: FloatArray
    left_toe_positions: FloatArray
    base_positions_smoothed: FloatArray
    right_ankle_positions_smoothed: FloatArray
    left_ankle_positions_smoothed: FloatArray
    right_toe_positions_smoothed: FloatArray
    left_toe_positions_smoothed: FloatArray
    right_floor_clearance: FloatArray
    left_floor_clearance: FloatArray
    base_velocity_smoothed: FloatArray
    right_toe_velocity_smoothed: FloatArray
    left_toe_velocity_smoothed: FloatArray
    right_source_label_raw: BoolArray | None
    left_source_label_raw: BoolArray | None
    right_source_label: BoolArray | None
    left_source_label: BoolArray | None
    right_contact_label: BoolArray
    left_contact_label: BoolArray
    right_confidence: FloatArray
    left_confidence: FloatArray
    flight_label: BoolArray | None
    right_contact_segments: IntArray
    left_contact_segments: IntArray

    def __post_init__(self) -> None:
        try:
            fps = float(self.fps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MotionValidationError("Contact FPS must be a finite positive scalar.") from exc
        if not np.isfinite(fps) or fps <= 0.0:
            raise MotionValidationError("Contact FPS must be a finite positive scalar.")
        object.__setattr__(self, "fps", fps)

        float_fields = (
            "seconds",
            "base_positions",
            "right_ankle_positions",
            "left_ankle_positions",
            "right_toe_positions",
            "left_toe_positions",
            "base_positions_smoothed",
            "right_ankle_positions_smoothed",
            "left_ankle_positions_smoothed",
            "right_toe_positions_smoothed",
            "left_toe_positions_smoothed",
            "right_floor_clearance",
            "left_floor_clearance",
            "base_velocity_smoothed",
            "right_toe_velocity_smoothed",
            "left_toe_velocity_smoothed",
            "right_confidence",
            "left_confidence",
        )
        for field_name in float_fields:
            object.__setattr__(
                self,
                field_name,
                _readonly_float(getattr(self, field_name)),
            )

        optional_bool_fields = (
            "right_source_label_raw",
            "left_source_label_raw",
            "right_source_label",
            "left_source_label",
            "flight_label",
        )
        for field_name in optional_bool_fields:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _readonly_bool(value))
        for field_name in ("right_contact_label", "left_contact_label"):
            object.__setattr__(self, field_name, _readonly_bool(getattr(self, field_name)))
        for field_name in ("right_contact_segments", "left_contact_segments"):
            object.__setattr__(self, field_name, _readonly_int(getattr(self, field_name)))

        frame_count = int(self.seconds.shape[0])
        if self.seconds.shape != (frame_count,):
            raise MotionValidationError("Contact seconds must be one-dimensional.")
        vector_fields = (
            "base_positions",
            "right_ankle_positions",
            "left_ankle_positions",
            "right_toe_positions",
            "left_toe_positions",
            "base_positions_smoothed",
            "right_ankle_positions_smoothed",
            "left_ankle_positions_smoothed",
            "right_toe_positions_smoothed",
            "left_toe_positions_smoothed",
            "base_velocity_smoothed",
            "right_toe_velocity_smoothed",
            "left_toe_velocity_smoothed",
        )
        for field_name in vector_fields:
            if getattr(self, field_name).shape != (frame_count, 3):
                raise MotionValidationError(
                    f"Contact field {field_name} must have shape ({frame_count}, 3)."
                )
        scalar_fields = (
            "right_floor_clearance",
            "left_floor_clearance",
            "right_confidence",
            "left_confidence",
            "right_contact_label",
            "left_contact_label",
        )
        for field_name in scalar_fields:
            if getattr(self, field_name).shape != (frame_count,):
                raise MotionValidationError(
                    f"Contact field {field_name} must have shape ({frame_count},)."
                )
        for field_name in optional_bool_fields:
            value = getattr(self, field_name)
            if value is not None and value.shape != (frame_count,):
                raise MotionValidationError(
                    f"Contact field {field_name} must have shape ({frame_count},)."
                )
        for field_name in ("right_contact_segments", "left_contact_segments"):
            segments = getattr(self, field_name)
            if segments.ndim != 2 or segments.shape[1:] != (2,):
                raise MotionValidationError(f"Contact field {field_name} must have shape (N, 2).")
            if np.any(segments < 0) or np.any(segments[:, 0] >= segments[:, 1]):
                raise MotionValidationError(f"Contact field {field_name} has invalid bounds.")
            if np.any(segments[:, 1] > frame_count):
                raise MotionValidationError(f"Contact field {field_name} exceeds the frame count.")

        for field_name in float_fields:
            if not np.isfinite(getattr(self, field_name)).all():
                raise MotionValidationError(f"Contact field {field_name} contains NaN or infinity.")
        for field_name in ("right_confidence", "left_confidence"):
            confidence = getattr(self, field_name)
            if np.any((confidence < 0.0) | (confidence > 1.0)):
                raise MotionValidationError(f"Contact field {field_name} must lie in [0, 1].")

    @property
    def frame_count(self) -> int:
        """Number of contact frames."""

        return int(self.seconds.shape[0])

    def reference_arrays(self) -> dict[str, NDArray[np.generic]]:
        """Return the Stage 1 archive arrays."""

        arrays: dict[str, NDArray[np.generic]] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "p_base_src_array": self.base_positions,
            "p_ra_src_array": self.right_ankle_positions,
            "p_la_src_array": self.left_ankle_positions,
            "p_rt_src_array": self.right_toe_positions,
            "p_lt_src_array": self.left_toe_positions,
            "p_base_src_smt_array": self.base_positions_smoothed,
            "p_ra_src_smt_array": self.right_ankle_positions_smoothed,
            "p_la_src_smt_array": self.left_ankle_positions_smoothed,
            "p_rt_src_smt_array": self.right_toe_positions_smoothed,
            "p_lt_src_smt_array": self.left_toe_positions_smoothed,
            "r_floor_clearance": self.right_floor_clearance,
            "l_floor_clearance": self.left_floor_clearance,
            "v_base_src_smt_array": self.base_velocity_smoothed,
            "v_rt_src_smt_array": self.right_toe_velocity_smoothed,
            "v_lt_src_smt_array": self.left_toe_velocity_smoothed,
        }
        optional_arrays = (
            ("r_contact_label_src_raw", self.right_source_label_raw),
            ("l_contact_label_src_raw", self.left_source_label_raw),
            ("r_contact_label_src", self.right_source_label),
            ("l_contact_label_src", self.left_source_label),
        )
        for name, value in optional_arrays:
            if value is not None:
                arrays[name] = value
        if self.right_source_label is not None:
            arrays["r_contact_label_merged"] = self.right_contact_label
        if self.left_source_label is not None:
            arrays["l_contact_label_merged"] = self.left_contact_label
        arrays["r_contact_confidence"] = self.right_confidence
        arrays["l_contact_confidence"] = self.left_confidence
        if self.flight_label is not None:
            arrays["flight_contact_label"] = self.flight_label
        arrays.update(
            {
                "right_contact_confidence": self.right_confidence,
                "right_contact_label": self.right_contact_label,
                "right_contact_segments": self.right_contact_segments,
                "left_contact_confidence": self.left_confidence,
                "left_contact_label": self.left_contact_label,
                "left_contact_segments": self.left_contact_segments,
            }
        )
        return arrays


def _rolling_quantile_points(
    seconds: FloatArray,
    values: FloatArray,
    *,
    window_seconds: float = 0.5,
    quantile: float = 0.05,
) -> tuple[FloatArray, FloatArray]:
    seconds = np.asarray(seconds).reshape(-1)
    values = np.asarray(values).reshape(-1)
    dt = float(seconds[1] - seconds[0])
    half = max(1, int(round(0.5 * window_seconds / dt)))
    xs: list[float] = []
    ys: list[float] = []
    for index in range(len(values)):
        lower = max(0, index - half)
        upper = min(len(values), index + half + 1)
        xs.append(float(seconds[index]))
        ys.append(float(np.quantile(values[lower:upper], quantile)))
    return np.asarray(xs), np.asarray(ys)


def _fit_line(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _gaussian_smooth(
    seconds: FloatArray,
    trajectory: FloatArray,
    *,
    smooth_time: float = 0.5,
    truncate: float = 3.0,
) -> FloatArray:
    seconds = np.asarray(seconds).reshape(-1)
    trajectory = np.asarray(trajectory)
    dt = float(seconds[1] - seconds[0])
    sigma = max(float(smooth_time) / max(dt, 1e-12), 1e-12)
    return np.asarray(
        gaussian_filter1d(
            trajectory,
            sigma=sigma,
            axis=0,
            mode="nearest",
            truncate=truncate,
        ),
        dtype=np.float64,
    )


def _consecutive_segments(indices: IntArray, minimum_length: int = 1) -> list[IntArray]:
    values = np.asarray(indices, dtype=int).reshape(-1)
    if len(values) == 0:
        return []
    segments: list[IntArray] = []
    start = 0
    for index in range(1, len(values)):
        if values[index] != values[index - 1] + 1:
            segment = values[start:index]
            if len(segment) >= minimum_length:
                segments.append(np.asarray(segment, dtype=np.int64))
            start = index
    segment = values[start:]
    if len(segment) >= minimum_length:
        segments.append(np.asarray(segment, dtype=np.int64))
    return segments


def _merge_close_segments(
    segments: list[IntArray],
    dt: float,
    *,
    minimum_gap_seconds: float = 0.05,
) -> list[IntArray]:
    if not segments:
        return []
    merged = [np.asarray(segments[0], dtype=np.int64)]
    maximum_gap_tick = int(round(minimum_gap_seconds / max(float(dt), 1e-12)))
    for source_segment in segments[1:]:
        segment = np.asarray(source_segment, dtype=np.int64)
        if int(segment[0]) - int(merged[-1][-1]) <= maximum_gap_tick:
            merged[-1] = np.arange(int(merged[-1][0]), int(segment[-1]) + 1, dtype=np.int64)
        else:
            merged.append(segment)
    return merged


def _binary_contact_median(
    label: BoolArray,
    sampling_hz: int,
    *,
    window_time: float = 5.0 / 60.0,
) -> BoolArray:
    label = np.asarray(label, dtype=bool).reshape(-1)
    window = max(1, int(round(float(window_time) * float(sampling_hz))))
    if window % 2 == 0:
        window += 1
    return np.asarray(
        median_filter(label.astype(np.uint8), size=window, mode="nearest").astype(bool),
        dtype=np.bool_,
    )


def _velocity_matrix_same_length(frame_count: int, dt: float) -> FloatArray:
    matrix = np.zeros((frame_count, frame_count), dtype=float)
    matrix[0, 0] = -1.0 / dt
    matrix[0, 1] = 1.0 / dt
    if frame_count == 2:
        matrix[1, 0] = -1.0 / dt
        matrix[1, 1] = 1.0 / dt
        return matrix
    for index in range(1, frame_count - 1):
        matrix[index, index - 1] = -0.5 / dt
        matrix[index, index + 1] = 0.5 / dt
    matrix[frame_count - 1, frame_count - 2] = -1.0 / dt
    matrix[frame_count - 1, frame_count - 1] = 1.0 / dt
    return matrix


def _segment_bounds(segments: list[IntArray]) -> IntArray:
    return np.asarray(
        [(int(segment[0]), int(segment[-1]) + 1) for segment in segments],
        dtype=np.int64,
    ).reshape(-1, 2)


def build_contact_schedule(
    motion: SomaMotion,
    *,
    source_joi: SomaJoiTrajectory | None = None,
) -> ContactSchedule:
    """Build a robot-independent toe-contact schedule.

    No stance or robot-relative normalization is applied.  ``source_joi`` may
    be supplied to reuse a trajectory already extracted for DMR.
    """

    if motion.frame_count < 2:
        raise MotionValidationError("Contact preprocessing requires at least two frames.")
    joi = extract_soma_joi(motion) if source_joi is None else source_joi
    if joi.frame_count != motion.frame_count:
        raise MotionValidationError("Source JOI frame count does not match the SOMA motion.")
    if not np.array_equal(joi.seconds, motion.seconds):
        raise MotionValidationError("Source JOI timestamps do not match the SOMA motion.")

    seconds = np.asarray(motion.seconds, dtype=np.float64)
    frame_count = motion.frame_count
    dt = float(seconds[1] - seconds[0])
    sampling_hz = int(1.0 / dt)

    floor_distance_threshold = 0.02
    smooth_time = 0.1
    velocity_threshold = 0.5
    flight_height_threshold = 0.045
    flight_run_xy_speed_threshold = 1.5
    flight_run_min_clearance_threshold = 0.025
    flight_run_toe_vz_threshold = 0.20
    flight_vertical_min_clearance_threshold = 0.03
    flight_vertical_toe_vz_threshold = 0.30
    maximum_contact_bridge_time = 0.20
    soft_double_support_time = 0.13
    flight_edge_ramp_time = 0.10
    flight_edge_run_xy_vz_ratio_threshold = 5.0
    soft_double_support_height_threshold = 0.05
    soft_double_support_toe_speed_threshold = 0.65
    minimum_geometry_segment_length = max(1, int(0.1 * sampling_hz))
    minimum_label_segment_length = max(1, int(0.03 * sampling_hz))

    base_positions = np.asarray(joi.positions("base"), dtype=np.float64)
    right_ankle_positions = np.asarray(joi.positions("ra"), dtype=np.float64)
    left_ankle_positions = np.asarray(joi.positions("la"), dtype=np.float64)
    right_toe_positions = np.asarray(joi.positions("rtoe"), dtype=np.float64)
    left_toe_positions = np.asarray(joi.positions("ltoe"), dtype=np.float64)

    right_floor_seconds, right_floor_points = _rolling_quantile_points(
        seconds, right_toe_positions[:, 2], window_seconds=0.5, quantile=0.05
    )
    left_floor_seconds, left_floor_points = _rolling_quantile_points(
        seconds, left_toe_positions[:, 2], window_seconds=0.5, quantile=0.05
    )
    right_floor_slope, right_floor_intercept = _fit_line(right_floor_seconds, right_floor_points)
    left_floor_slope, left_floor_intercept = _fit_line(left_floor_seconds, left_floor_points)
    right_floor_line = right_floor_slope * seconds + right_floor_intercept
    left_floor_line = left_floor_slope * seconds + left_floor_intercept
    right_floor_clearance = right_toe_positions[:, 2] - right_floor_line
    left_floor_clearance = left_toe_positions[:, 2] - left_floor_line
    right_close = right_floor_clearance <= floor_distance_threshold
    left_close = left_floor_clearance <= floor_distance_threshold

    base_positions_smoothed = _gaussian_smooth(seconds, base_positions, smooth_time=smooth_time)
    right_ankle_positions_smoothed = _gaussian_smooth(
        seconds, right_ankle_positions, smooth_time=smooth_time
    )
    left_ankle_positions_smoothed = _gaussian_smooth(
        seconds, left_ankle_positions, smooth_time=smooth_time
    )
    right_toe_positions_smoothed = _gaussian_smooth(
        seconds, right_toe_positions, smooth_time=smooth_time
    )
    left_toe_positions_smoothed = _gaussian_smooth(
        seconds, left_toe_positions, smooth_time=smooth_time
    )
    velocity_matrix = _velocity_matrix_same_length(frame_count, dt)
    right_toe_velocity_smoothed = velocity_matrix @ right_toe_positions_smoothed
    left_toe_velocity_smoothed = velocity_matrix @ left_toe_positions_smoothed
    base_velocity_smoothed = velocity_matrix @ base_positions_smoothed
    base_xy_speed_smoothed = np.linalg.norm(base_velocity_smoothed[:, :2], axis=1)
    right_mean_abs_velocity = np.mean(np.abs(right_toe_velocity_smoothed), axis=1)
    left_mean_abs_velocity = np.mean(np.abs(left_toe_velocity_smoothed), axis=1)
    right_slow = np.abs(right_mean_abs_velocity) <= velocity_threshold
    left_slow = np.abs(left_mean_abs_velocity) <= velocity_threshold

    right_geometry_segments = _merge_close_segments(
        _consecutive_segments(
            np.nonzero(right_close & right_slow)[0], minimum_geometry_segment_length
        ),
        dt,
        minimum_gap_seconds=0.2,
    )
    left_geometry_segments = _merge_close_segments(
        _consecutive_segments(
            np.nonzero(left_close & left_slow)[0], minimum_geometry_segment_length
        ),
        dt,
        minimum_gap_seconds=0.2,
    )

    contact_source = "geometry_fallback_toe"
    left_label: BoolArray | None = None
    right_label: BoolArray | None = None
    if motion.foot_contacts is not None and motion.foot_contacts.shape[0] == frame_count:
        contacts = np.asarray(motion.foot_contacts).astype(bool)
        if contacts.shape[1] >= 6:
            left_label = np.any(contacts[:, 1:3], axis=1)
            right_label = np.any(contacts[:, 4:6], axis=1)
            contact_source = "kimodo_toe_contacts_6ch"
        elif contacts.shape[1] >= 4:
            left_label = contacts[:, 1]
            right_label = contacts[:, 3]
            contact_source = "kimodo_toe_contacts_4ch"

    if left_label is not None and right_label is not None:
        left_source_label_raw = left_label.copy()
        right_source_label_raw = right_label.copy()
        left_label = _binary_contact_median(left_source_label_raw, sampling_hz)
        right_label = _binary_contact_median(right_source_label_raw, sampling_hz)
        left_source_label = left_label.copy()
        right_source_label = right_label.copy()
        source_both_toes_off = ~(left_source_label | right_source_label)

        left_geometry_supplement = source_both_toes_off & left_close & left_slow
        right_geometry_supplement = source_both_toes_off & right_close & right_slow
        both_geometry_support = left_geometry_supplement & right_geometry_supplement
        for gap_tick in np.nonzero(both_geometry_support)[0]:
            right_support_cost = max(float(right_floor_clearance[gap_tick]), 0.0) / max(
                floor_distance_threshold, 1e-6
            )
            left_support_cost = max(float(left_floor_clearance[gap_tick]), 0.0) / max(
                floor_distance_threshold, 1e-6
            )
            right_support_cost += abs(float(right_toe_velocity_smoothed[gap_tick, 2])) / max(
                flight_run_toe_vz_threshold, 1e-6
            )
            left_support_cost += abs(float(left_toe_velocity_smoothed[gap_tick, 2])) / max(
                flight_run_toe_vz_threshold, 1e-6
            )
            if right_support_cost <= left_support_cost:
                left_geometry_supplement[gap_tick] = False
            else:
                right_geometry_supplement[gap_tick] = False
        left_label |= left_geometry_supplement
        right_label |= right_geometry_supplement

        flight_label = np.zeros(frame_count, dtype=bool)
        left_bridge_supplement = np.zeros(frame_count, dtype=bool)
        right_bridge_supplement = np.zeros(frame_count, dtype=bool)
        both_off_after_geometry = ~(left_label | right_label)
        remaining_gap_segments = _consecutive_segments(np.nonzero(both_off_after_geometry)[0], 1)
        for gap_segment in remaining_gap_segments:
            gap_segment = np.asarray(gap_segment, dtype=int)
            lower_is_right = right_floor_clearance[gap_segment] <= left_floor_clearance[gap_segment]
            lower_clearance = np.minimum(
                right_floor_clearance[gap_segment], left_floor_clearance[gap_segment]
            )
            lower_toe_vz = np.where(
                lower_is_right,
                right_toe_velocity_smoothed[gap_segment, 2],
                left_toe_velocity_smoothed[gap_segment, 2],
            )
            median_clearance = float(np.median(lower_clearance))
            median_base_xy_speed = float(np.median(base_xy_speed_smoothed[gap_segment]))
            median_lower_toe_vz = float(np.median(np.abs(lower_toe_vz)))
            high_clearance_flight = median_clearance >= flight_height_threshold
            running_flight = (
                median_base_xy_speed >= flight_run_xy_speed_threshold
                and median_clearance >= flight_run_min_clearance_threshold
                and median_lower_toe_vz >= flight_run_toe_vz_threshold
            )
            vertical_flight = (
                median_clearance >= flight_vertical_min_clearance_threshold
                and median_lower_toe_vz >= flight_vertical_toe_vz_threshold
            )
            if high_clearance_flight or running_flight or vertical_flight:
                flight_label[gap_segment] = True
                continue
            if len(gap_segment) * dt > maximum_contact_bridge_time:
                continue
            previous_tick = int(gap_segment[0]) - 1
            next_tick = int(gap_segment[-1]) + 1
            previous_right = previous_tick >= 0 and bool(right_label[previous_tick])
            previous_left = previous_tick >= 0 and bool(left_label[previous_tick])
            next_right = next_tick < frame_count and bool(right_label[next_tick])
            next_left = next_tick < frame_count and bool(left_label[next_tick])
            gap_middle = (len(gap_segment) - 1) / 2.0
            for local_index, gap_tick in enumerate(gap_segment):
                use_previous = local_index <= gap_middle
                use_next = not use_previous
                add_right = (use_previous and previous_right) or (use_next and next_right)
                add_left = (use_previous and previous_left) or (use_next and next_left)
                if not (add_right or add_left):
                    right_support_cost = max(float(right_floor_clearance[gap_tick]), 0.0) / max(
                        floor_distance_threshold, 1e-6
                    )
                    left_support_cost = max(float(left_floor_clearance[gap_tick]), 0.0) / max(
                        floor_distance_threshold, 1e-6
                    )
                    right_support_cost += abs(
                        float(right_toe_velocity_smoothed[gap_tick, 2])
                    ) / max(flight_run_toe_vz_threshold, 1e-6)
                    left_support_cost += abs(float(left_toe_velocity_smoothed[gap_tick, 2])) / max(
                        flight_run_toe_vz_threshold, 1e-6
                    )
                    add_right = right_support_cost <= left_support_cost
                    add_left = not add_right
                right_bridge_supplement[gap_tick] = add_right
                left_bridge_supplement[gap_tick] = add_left

        right_label |= right_bridge_supplement
        left_label |= left_bridge_supplement
        contact_source += "+geometry_gap_fill+flight_aware_bridge"

        right_merged_segments = _merge_close_segments(
            _consecutive_segments(np.nonzero(right_label)[0], minimum_label_segment_length),
            dt,
            minimum_gap_seconds=0.05,
        )
        left_merged_segments = _merge_close_segments(
            _consecutive_segments(np.nonzero(left_label)[0], minimum_label_segment_length),
            dt,
            minimum_gap_seconds=0.05,
        )
        right_contact_label = np.zeros(frame_count, dtype=bool)
        left_contact_label = np.zeros(frame_count, dtype=bool)
        for segment in right_merged_segments:
            right_contact_label[np.asarray(segment, dtype=int)] = True
        for segment in left_merged_segments:
            left_contact_label[np.asarray(segment, dtype=int)] = True
        right_contact_label[flight_label] = False
        left_contact_label[flight_label] = False
        right_segments = _consecutive_segments(
            np.nonzero(right_contact_label)[0], minimum_label_segment_length
        )
        left_segments = _consecutive_segments(
            np.nonzero(left_contact_label)[0], minimum_label_segment_length
        )

        right_confidence = right_contact_label.astype(np.float64)
        left_confidence = left_contact_label.astype(np.float64)
        right_toe_speed = np.linalg.norm(right_toe_velocity_smoothed, axis=1)
        left_toe_speed = np.linalg.norm(left_toe_velocity_smoothed, axis=1)
        number_soft_frames = max(1, int(round(soft_double_support_time * sampling_hz)))
        number_flight_edge_frames = max(1, int(round(flight_edge_ramp_time * sampling_hz)))

        for switch_tick in range(1, frame_count):
            previous_right = bool(right_contact_label[switch_tick - 1])
            previous_left = bool(left_contact_label[switch_tick - 1])
            current_right = bool(right_contact_label[switch_tick])
            current_left = bool(left_contact_label[switch_tick])
            left_to_right = (
                previous_left and not previous_right and current_right and not current_left
            )
            right_to_left = (
                previous_right and not previous_left and current_left and not current_right
            )
            if not (left_to_right or right_to_left):
                continue

            bridge_mask = right_bridge_supplement | left_bridge_supplement
            bridge_handover = bool(bridge_mask[switch_tick - 1] and bridge_mask[switch_tick])
            if bridge_handover:
                handover_stop = min(frame_count, switch_tick + number_soft_frames)
                transition_ticks = np.arange(switch_tick, handover_stop, dtype=int)
                invalid_offsets = np.nonzero(
                    ~source_both_toes_off[transition_ticks] | flight_label[transition_ticks]
                )[0]
                if len(invalid_offsets) > 0:
                    transition_ticks = np.arange(
                        switch_tick,
                        switch_tick + int(invalid_offsets[0]),
                        dtype=int,
                    )
                for local_index, transition_tick in enumerate(transition_ticks):
                    alpha = float(local_index + 1) / float(len(transition_ticks) + 1)
                    if left_to_right:
                        left_confidence[transition_tick] = 1.0 - alpha
                        right_confidence[transition_tick] = alpha
                    else:
                        right_confidence[transition_tick] = 1.0 - alpha
                        left_confidence[transition_tick] = alpha
                continue

            stop_tick = min(frame_count, switch_tick + number_soft_frames)
            transition_ticks = np.arange(switch_tick, stop_tick, dtype=int)
            hard_double_offsets = np.nonzero(
                right_contact_label[transition_ticks] & left_contact_label[transition_ticks]
            )[0]
            if len(hard_double_offsets) > 0:
                transition_ticks = np.arange(
                    switch_tick,
                    switch_tick + int(hard_double_offsets[0]),
                    dtype=int,
                )
            if len(transition_ticks) == 0:
                continue
            if left_to_right:
                incoming_stable = np.all(right_contact_label[transition_ticks])
                outgoing_plausible = (
                    left_floor_clearance[transition_ticks] <= soft_double_support_height_threshold
                ) & (left_toe_speed[transition_ticks] <= soft_double_support_toe_speed_threshold)
            else:
                incoming_stable = np.all(left_contact_label[transition_ticks])
                outgoing_plausible = (
                    right_floor_clearance[transition_ticks] <= soft_double_support_height_threshold
                ) & (right_toe_speed[transition_ticks] <= soft_double_support_toe_speed_threshold)
            if not incoming_stable or float(np.mean(outgoing_plausible)) < 0.75:
                continue
            for local_index, transition_tick in enumerate(transition_ticks):
                if not outgoing_plausible[local_index]:
                    continue
                alpha = float(local_index + 1) / float(len(transition_ticks) + 1)
                if left_to_right:
                    left_confidence[transition_tick] = max(
                        left_confidence[transition_tick], 1.0 - alpha
                    )
                    right_confidence[transition_tick] = min(
                        right_confidence[transition_tick], alpha
                    )
                else:
                    right_confidence[transition_tick] = max(
                        right_confidence[transition_tick], 1.0 - alpha
                    )
                    left_confidence[transition_tick] = min(left_confidence[transition_tick], alpha)

        flight_segments = _consecutive_segments(np.nonzero(flight_label)[0], 1)
        for flight_segment in flight_segments:
            flight_segment = np.asarray(flight_segment, dtype=int)
            flight_start = int(flight_segment[0])
            flight_end = int(flight_segment[-1])
            lower_is_right = (
                right_floor_clearance[flight_segment] <= left_floor_clearance[flight_segment]
            )
            lower_toe_vz = np.where(
                lower_is_right,
                right_toe_velocity_smoothed[flight_segment, 2],
                left_toe_velocity_smoothed[flight_segment, 2],
            )
            flight_xy_speed = float(np.median(base_xy_speed_smoothed[flight_segment]))
            flight_toe_abs_vz = float(np.median(np.abs(lower_toe_vz)))
            flight_xy_vz_ratio = flight_xy_speed / max(flight_toe_abs_vz, 1e-6)
            if (
                flight_xy_speed < flight_run_xy_speed_threshold
                or flight_xy_vz_ratio < flight_edge_run_xy_vz_ratio_threshold
            ):
                continue
            pre_ticks = np.arange(
                max(0, flight_start - number_flight_edge_frames),
                flight_start,
                dtype=int,
            )
            post_ticks = np.arange(
                flight_end + 1,
                min(frame_count, flight_end + 1 + number_flight_edge_frames),
                dtype=int,
            )
            pre_weights = np.linspace(1.0, 0.0, len(pre_ticks) + 2)[1:-1]
            post_weights = np.linspace(0.0, 1.0, len(post_ticks) + 2)[1:-1]
            for edge_tick, edge_weight in zip(pre_ticks, pre_weights, strict=True):
                if right_contact_label[edge_tick]:
                    right_confidence[edge_tick] = min(right_confidence[edge_tick], edge_weight)
                if left_contact_label[edge_tick]:
                    left_confidence[edge_tick] = min(left_confidence[edge_tick], edge_weight)
            for edge_tick, edge_weight in zip(post_ticks, post_weights, strict=True):
                if right_contact_label[edge_tick]:
                    right_confidence[edge_tick] = min(right_confidence[edge_tick], edge_weight)
                if left_contact_label[edge_tick]:
                    left_confidence[edge_tick] = min(left_confidence[edge_tick], edge_weight)
    else:
        left_source_label_raw = None
        right_source_label_raw = None
        left_source_label = None
        right_source_label = None
        flight_label = None
        right_segments = right_geometry_segments
        left_segments = left_geometry_segments
        right_confidence = np.zeros(frame_count, dtype=np.float64)
        left_confidence = np.zeros(frame_count, dtype=np.float64)
        for segment in right_segments:
            right_confidence[np.asarray(segment, dtype=int)] = 1.0
        for segment in left_segments:
            left_confidence[np.asarray(segment, dtype=int)] = 1.0
        right_contact_label = np.zeros(frame_count, dtype=bool)
        left_contact_label = np.zeros(frame_count, dtype=bool)
        right_contact_label[:] = right_confidence >= 0.5
        left_contact_label[:] = left_confidence >= 0.5

    return ContactSchedule(
        fps=motion.fps,
        seconds=seconds,
        contact_source=contact_source,
        base_positions=base_positions,
        right_ankle_positions=right_ankle_positions,
        left_ankle_positions=left_ankle_positions,
        right_toe_positions=right_toe_positions,
        left_toe_positions=left_toe_positions,
        base_positions_smoothed=base_positions_smoothed,
        right_ankle_positions_smoothed=right_ankle_positions_smoothed,
        left_ankle_positions_smoothed=left_ankle_positions_smoothed,
        right_toe_positions_smoothed=right_toe_positions_smoothed,
        left_toe_positions_smoothed=left_toe_positions_smoothed,
        right_floor_clearance=right_floor_clearance,
        left_floor_clearance=left_floor_clearance,
        base_velocity_smoothed=base_velocity_smoothed,
        right_toe_velocity_smoothed=right_toe_velocity_smoothed,
        left_toe_velocity_smoothed=left_toe_velocity_smoothed,
        right_source_label_raw=right_source_label_raw,
        left_source_label_raw=left_source_label_raw,
        right_source_label=right_source_label,
        left_source_label=left_source_label,
        right_contact_label=right_contact_label,
        left_contact_label=left_contact_label,
        right_confidence=right_confidence,
        left_confidence=left_confidence,
        flight_label=flight_label,
        right_contact_segments=_segment_bounds(right_segments),
        left_contact_segments=_segment_bounds(left_segments),
    )
