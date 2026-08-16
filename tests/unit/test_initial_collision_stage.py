from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import rimkit.stages.initial_collision as collision_stage
from rimkit.exceptions import ConfigurationError, MotionValidationError
from rimkit.stages import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
    InitialCollisionResult,
)

ARCHIVE_KEYS = (
    "schema_version",
    "fps",
    "seconds",
    "qpos_cc_smt_array",
)


def _diagnostics() -> InitialCollisionDiagnostics:
    return InitialCollisionDiagnostics(
        backend="python_reference_ordered",
        distance_backend="mujoco_signed_distance",
        ik_backend="python_body_position_ik_reference_ordered",
        trajectory_backend="cvxpy_clarabel_with_fallback",
        root_geom_count=37,
        collision_geom_count=38,
        raw_candidate_pair_count=647,
        candidate_pair_count=375,
        arm_joint_names=("left_shoulder_pitch_joint",),
        input_violations=2,
        input_max_frame_violations=1,
        output_violations=0,
        output_max_frame_violations=0,
        passes=(
            CollisionPassDiagnostics(
                margin=0.02,
                violations=0,
                max_frame_violations=0,
            ),
        ),
    )


def test_initial_collision_result_owns_read_only_arrays_and_exact_archive_contract() -> None:
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float32)
    qpos = np.arange(72, dtype=np.float32).reshape(2, 36)
    expected_seconds = seconds.astype(np.float64)
    expected_qpos = qpos.astype(np.float64)

    result = InitialCollisionResult(
        robot_id="g1",
        fps=30.0,
        seconds=seconds,
        qpos=qpos,
        diagnostics=_diagnostics(),
    )
    seconds[:] = -1.0
    qpos[:] = -1.0

    assert result.robot_id == "g1"
    assert result.fps == 30.0
    assert result.seconds.dtype == np.dtype(np.float64)
    assert result.qpos.dtype == np.dtype(np.float64)
    assert not result.seconds.flags.writeable
    assert not result.qpos.flags.writeable
    np.testing.assert_array_equal(result.seconds, expected_seconds)
    np.testing.assert_array_equal(result.qpos, expected_qpos)
    with pytest.raises(ValueError):
        result.seconds[0] = 1.0
    with pytest.raises(ValueError):
        result.qpos[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.fps = 60.0  # type: ignore[misc]

    archive = result.reference_arrays()
    assert tuple(archive) == ARCHIVE_KEYS
    assert archive["schema_version"].shape == ()
    assert archive["schema_version"].dtype == np.dtype(np.int32)
    assert int(archive["schema_version"]) == 1
    assert archive["fps"].shape == ()
    assert archive["fps"].dtype == np.dtype(np.float64)
    assert float(archive["fps"]) == 30.0
    assert archive["seconds"] is result.seconds
    assert archive["qpos_cc_smt_array"] is result.qpos


@pytest.mark.parametrize(
    ("seconds", "qpos", "fps", "message"),
    (
        (np.asarray([0.0]), np.zeros(36), 30.0, "shape"),
        (np.asarray([0.0, 0.1]), np.zeros((1, 36)), 30.0, "shape"),
        (np.asarray([0.0]), np.full((1, 36), np.nan), 30.0, "finite"),
        (np.asarray([0.0]), np.zeros((1, 36)), 0.0, "positive"),
        (np.asarray([0.0]), np.zeros((1, 36)), np.nan, "positive"),
    ),
)
def test_initial_collision_result_validates_shape_finiteness_and_fps(
    seconds: np.ndarray,
    qpos: np.ndarray,
    fps: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        InitialCollisionResult(
            robot_id="g1",
            fps=fps,
            seconds=seconds,
            qpos=qpos,
            diagnostics=_diagnostics(),
        )


@pytest.mark.parametrize(
    ("qpos", "seconds", "fps", "message"),
    (
        (np.zeros((2, 35)), np.asarray([0.0, 0.1]), 30.0, "shape"),
        (np.empty((0, 36)), np.empty(0), 30.0, "at least one frame"),
        (np.full((2, 36), np.nan), np.asarray([0.0, 0.1]), 30.0, "NaN or infinity"),
        (np.zeros((2, 36)), np.asarray([0.0, 0.0]), 30.0, "strictly increasing"),
        (np.zeros((2, 36)), np.asarray([0.0, 0.1]), 0.0, "positive"),
        (np.zeros((2, 36)), np.asarray([0.0, 0.1]), np.inf, "positive"),
    ),
)
def test_run_initial_collision_validates_public_stage_inputs_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    qpos: np.ndarray,
    seconds: np.ndarray,
    fps: float,
    message: str,
) -> None:
    monkeypatch.setattr(collision_stage, "_validate_reference_environment", lambda: None)

    with pytest.raises(MotionValidationError, match=message):
        collision_stage.run_initial_collision(
            qpos,
            seconds,
            robot_id="g1",
            fps=fps,
        )


def test_stage3_rejects_a_mujoco_version_other_than_the_frozen_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collision_stage.mujoco, "__version__", "3.5.0")

    with pytest.raises(ConfigurationError, match=r"MuJoCo 3\.6\.0.*3\.5\.0"):
        collision_stage.run_initial_collision(
            np.zeros((1, 36), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            robot_id="g1",
            fps=30.0,
        )


def test_stage3_accepts_the_frozen_reference_mujoco_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collision_stage.mujoco, "__version__", "3.6.0")

    collision_stage._validate_reference_environment()
