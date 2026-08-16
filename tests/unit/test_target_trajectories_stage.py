from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import rimkit.stages.target_trajectories as target_stage
from rimkit.exceptions import MotionValidationError
from rimkit.stages import TargetTrajectoriesResult

ARCHIVE_KEYS = (
    "schema_version",
    "fps",
    "seconds",
    "p_root_trgt_array",
    "p_ra_trgt_array",
    "p_la_trgt_array",
    "p_rt_trgt_array",
    "p_lt_trgt_array",
    "p_root_trgt_smt_array",
    "p_ra_trgt_smt_array",
    "p_la_trgt_smt_array",
    "p_rt_trgt_smt_array",
    "p_lt_trgt_smt_array",
)


def _result() -> TargetTrajectoriesResult:
    arrays = [np.full((2, 3), index, dtype=np.float32) for index in range(10)]
    return TargetTrajectoriesResult(
        robot_id="g1",
        fps=30.0,
        seconds=np.asarray([0.0, 1.0 / 30.0], dtype=np.float32),
        root=arrays[0],
        right_ankle=arrays[1],
        left_ankle=arrays[2],
        right_toe=arrays[3],
        left_toe=arrays[4],
        root_smoothed=arrays[5],
        right_ankle_smoothed=arrays[6],
        left_ankle_smoothed=arrays[7],
        right_toe_smoothed=arrays[8],
        left_toe_smoothed=arrays[9],
    )


def test_target_trajectory_result_owns_read_only_arrays_and_exact_contract() -> None:
    result = _result()

    assert result.seconds.dtype == np.dtype(np.float64)
    assert result.root.dtype == np.dtype(np.float64)
    assert not result.seconds.flags.writeable
    assert not result.root.flags.writeable
    assert result.frame_count == 2
    with pytest.raises(ValueError):
        result.root[0, 0] = 5.0
    with pytest.raises(FrozenInstanceError):
        result.fps = 60.0  # type: ignore[misc]

    archive = result.reference_arrays()
    assert tuple(archive) == ARCHIVE_KEYS
    assert archive["schema_version"].dtype == np.dtype(np.int32)
    assert archive["fps"].dtype == np.dtype(np.float64)
    assert archive["seconds"] is result.seconds
    assert archive["p_root_trgt_array"] is result.root
    assert archive["p_lt_trgt_smt_array"] is result.left_toe_smoothed


@pytest.mark.parametrize(
    ("seconds", "trajectory", "fps", "message"),
    (
        (np.empty(0), np.empty((0, 3)), 30.0, "at least one frame"),
        (np.asarray([0.0, 0.1]), np.empty((1, 3)), 30.0, "shape"),
        (np.asarray([0.0]), np.full((1, 3), np.nan), 30.0, "NaN or infinity"),
        (np.asarray([0.0]), np.zeros((1, 3)), 0.0, "positive"),
    ),
)
def test_target_trajectory_result_validates_contract(
    seconds: np.ndarray,
    trajectory: np.ndarray,
    fps: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TargetTrajectoriesResult(
            robot_id="g1",
            fps=fps,
            seconds=seconds,
            root=trajectory,
            right_ankle=trajectory,
            left_ankle=trajectory,
            right_toe=trajectory,
            left_toe=trajectory,
            root_smoothed=trajectory,
            right_ankle_smoothed=trajectory,
            left_ankle_smoothed=trajectory,
            right_toe_smoothed=trajectory,
            left_toe_smoothed=trajectory,
        )


@pytest.mark.parametrize(
    ("raw", "smoothed", "seconds", "fps", "message"),
    (
        (np.zeros((2, 35)), np.zeros((2, 36)), np.asarray([0.0, 0.1]), 30.0, "DMR"),
        (
            np.zeros((2, 36)),
            np.zeros((2, 35)),
            np.asarray([0.0, 0.1]),
            30.0,
            "collision-refined",
        ),
        (np.empty((0, 36)), np.empty((0, 36)), np.empty(0), 30.0, "at least one"),
        (
            np.full((2, 36), np.nan),
            np.zeros((2, 36)),
            np.asarray([0.0, 0.1]),
            30.0,
            "NaN or infinity",
        ),
        (
            np.zeros((2, 36)),
            np.zeros((2, 36)),
            np.asarray([0.0, 0.0]),
            30.0,
            "strictly increasing",
        ),
        (
            np.zeros((2, 36)),
            np.zeros((2, 36)),
            np.asarray([0.0, 0.1]),
            0.0,
            "positive",
        ),
    ),
)
def test_run_target_trajectories_validates_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    raw: np.ndarray,
    smoothed: np.ndarray,
    seconds: np.ndarray,
    fps: float,
    message: str,
) -> None:
    monkeypatch.setattr(
        target_stage.MujocoModel,
        "from_robot",
        lambda robot_id: pytest.fail(f"loaded model for invalid input: {robot_id}"),
    )
    with pytest.raises(MotionValidationError, match=message):
        target_stage.run_target_trajectories(
            raw,
            smoothed,
            seconds,
            robot_id="g1",
            fps=fps,
        )
