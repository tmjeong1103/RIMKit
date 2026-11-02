"""Production CATO macro partition, shared with the review renderer.

This is the existing penalized piecewise-constant implementation moved without
changing arithmetic, features, parameters, frame/N normalization or tie breaks.
Foot-specific intended-contact intervals remain separate from this partition.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

from rimkit.exceptions import MotionValidationError

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
CATO_SEGMENT_ALGORITHM = "penalized_piecewise_constant_cato_macro_v1"

_LOWER_BODY_JOINT_TOKENS = ("hip", "knee", "ankle", "toe")
_MACRO_SEGMENT_CONFIG = {
    "root_joint_smooth_time": 0.25,
    "contact_smooth_time": 0.35,
    "min_segment_time": 0.40,
    "max_segments": 8,
    "split_penalty_ratio": 0.08,
}


def lower_body_qpos_indices(robot_id: str) -> NDArray[np.int32]:
    from rimkit.mujoco.model import MujocoModel

    model = MujocoModel.from_robot(robot_id)
    names = tuple(
        name
        for name in model.rev_pri_joint_names
        if any(token in name.lower() for token in _LOWER_BODY_JOINT_TOKENS)
    )
    if not names:
        raise MotionValidationError(f"Robot {robot_id!r} has no CATO lower-body joints")
    return model.get_qpos_indices(names)


def build_cato_macro_segments(
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
    return ranges, cato_segment_phases(frame_boundaries, frame_count)


def cato_segment_phases(frame_boundaries: ArrayLike, frame_count: int) -> FloatArray:
    """Return segment boundary frames as phases rounded to 8 decimals.

    The builder and the export validator both use this, so they always agree;
    Python's round() and np.round() differ on some ties, e.g. at 2560 frames.
    """
    return np.round(np.asarray(frame_boundaries, dtype=np.float64) / float(frame_count), 8)


def cato_segment_provenance(*, contact_source: str, source_motion_sha256: str) -> dict[str, Any]:
    """Record the exact implementation and configuration; never approve labels."""
    repository = Path(__file__).resolve().parents[3]
    try:
        # An installed copy is not a checkout, and may sit inside some other
        # repository's virtual environment.
        if not (repository / ".git").exists():
            raise OSError("RIMKit is not running from a git checkout")
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {
        "algorithm": CATO_SEGMENT_ALGORITHM,
        "algorithm_version": 1,
        "config": dict(_MACRO_SEGMENT_CONFIG),
        "feature_names": [
            "root_xy_speed",
            "root_abs_z_speed",
            "lower_body_speed_rms",
            "contact_transition_rate",
            "flight_density",
        ],
        "feature_weights": [1.0, 0.35, 0.50, 0.20, 0.80],
        "feature_normalization": {"quantile": 0.90, "clip": [0.0, 2.0]},
        "smoothing": {"filter": "gaussian_filter1d", "mode": "nearest", "truncate": 3.0},
        "boundary_normalization": "round(frame_index / frame_count, 8)",
        "segment_frame_convention": "half_open_start_inclusive_end_exclusive",
        "contact_source": contact_source,
        "source_motion_sha256": source_motion_sha256,
        "rimkit_git_commit": commit,
        "rimkit_worktree_dirty": dirty,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def validate_cato_segments(
    ranges: ArrayLike, boundaries: ArrayLike, frame_count: int
) -> tuple[IntArray, FloatArray]:
    """Check that production metadata is a contiguous, frame-aligned partition."""
    raw = np.asarray(ranges)
    if raw.dtype.kind not in "iu" or raw.ndim != 2 or raw.shape[1:] != (2,) or len(raw) == 0:
        raise MotionValidationError("CATO segment frames must be nonempty integer (M, 2) ranges")
    frames = np.array(raw, dtype=np.int64, copy=True)
    phases = np.array(boundaries, dtype=np.float64, copy=True)
    if (
        phases.shape != (len(frames) + 1,)
        or not np.isfinite(phases).all()
        or phases[0] != 0
        or phases[-1] != 1
        or np.any(np.diff(phases) <= 0)
        or frames[0, 0] != 0
        or frames[-1, 1] != frame_count
        or np.any(frames[:, 0] >= frames[:, 1])
        or np.any(frames[1:, 0] != frames[:-1, 1])
    ):
        raise MotionValidationError("CATO segments must partition all original frames from 0 to 1")
    expected = cato_segment_phases(np.r_[frames[:, 0], frame_count], frame_count)
    if not np.array_equal(phases, expected):
        raise MotionValidationError("CATO segments differ from their frame/N boundaries")
    return frames, phases
