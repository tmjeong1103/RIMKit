from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import rimkit.stages.final_collision as final_stage
from rimkit.exceptions import ConfigurationError, MotionValidationError
from rimkit.stages.final_collision import (
    FinalCollisionDiagnostics,
    FinalCollisionResult,
)
from rimkit.stages.initial_collision import (
    CollisionPassDiagnostics,
    InitialCollisionDiagnostics,
    InitialCollisionResult,
)

QPOS_ONLY_KEYS = (
    "schema_version",
    "fps",
    "seconds",
    "qpos_cc_fpa_array",
)
FULL_REFERENCE_KEYS = QPOS_ONLY_KEYS + (
    "r_contact_label_merged",
    "l_contact_label_merged",
    "r_contact_confidence",
    "l_contact_confidence",
    "flight_contact_label",
    "right_contact_confidence",
    "right_contact_label",
    "right_contact_segments",
    "left_contact_confidence",
    "left_contact_label",
    "left_contact_segments",
)


def _initial_diagnostics() -> InitialCollisionDiagnostics:
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


def _final_diagnostics() -> FinalCollisionDiagnostics:
    return FinalCollisionDiagnostics.from_initial(_initial_diagnostics())


def test_final_collision_result_owns_arrays_and_reproduces_stage8_contact_contract() -> None:
    seconds = np.arange(5, dtype=np.float32) / 30.0
    qpos = np.arange(5 * 36, dtype=np.float32).reshape(5, 36)
    labels = np.asarray(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 0],
        ],
        dtype=np.int8,
    )
    confidence = labels.astype(np.float32)
    expected_seconds = seconds.astype(np.float64)
    expected_qpos = qpos.astype(np.float64)

    result = FinalCollisionResult(
        robot_id="G1",
        fps=30.0,
        seconds=seconds,
        qpos=qpos,
        diagnostics=_final_diagnostics(),
        contact_labels=labels,
        contact_confidence=confidence,
    )
    seconds[:] = -1.0
    qpos[:] = -1.0
    labels[:] = 0
    confidence[:] = 0.0

    assert result.robot_id == "g1"
    assert result.contact_labels is not None
    assert result.contact_confidence is not None
    assert result.flight_labels is not None
    assert result.contact_labels.shape == (5, 4)
    assert result.contact_confidence.shape == (5, 4)
    assert not result.seconds.flags.writeable
    assert not result.qpos.flags.writeable
    assert not result.contact_labels.flags.writeable
    np.testing.assert_array_equal(result.seconds, expected_seconds)
    np.testing.assert_array_equal(result.qpos, expected_qpos)
    np.testing.assert_array_equal(result.contact_labels[:, 2:], False)
    np.testing.assert_array_equal(result.contact_confidence[:, 2:], 0.0)
    np.testing.assert_array_equal(
        result.flight_labels,
        np.asarray([False, False, False, True, False]),
    )

    arrays = result.reference_arrays()
    assert tuple(arrays) == FULL_REFERENCE_KEYS
    np.testing.assert_array_equal(
        arrays["left_contact_segments"],
        np.asarray([[0, 2], [4, 5]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["right_contact_segments"],
        np.asarray([[1, 3]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["l_contact_label_merged"],
        arrays["left_contact_label"],
    )
    np.testing.assert_array_equal(
        arrays["r_contact_confidence"],
        arrays["right_contact_confidence"],
    )


def test_final_collision_result_without_contacts_keeps_qpos_only_contract() -> None:
    result = FinalCollisionResult(
        robot_id="h1",
        fps=30.0,
        seconds=np.asarray([0.0]),
        qpos=np.zeros((1, 27)),
        diagnostics=_final_diagnostics(),
    )

    assert tuple(result.reference_arrays()) == QPOS_ONLY_KEYS


def test_final_collision_diagnostics_are_copied_into_a_distinct_stage_type() -> None:
    initial = _initial_diagnostics()
    final = FinalCollisionDiagnostics.from_initial(initial)

    assert not isinstance(final, InitialCollisionDiagnostics)
    assert final.outer_passes == initial.outer_passes
    assert final.arm_joint_names == initial.arm_joint_names
    assert final.input_violations == initial.input_violations
    assert final.passes == initial.passes


def test_run_final_collision_forwards_validated_stage7_arrays_to_shared_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seconds = np.asarray([0.0, 1.0 / 30.0])
    qpos = np.zeros((2, 30), dtype=np.float32)
    labels = np.zeros((2, 4), dtype=bool)
    confidence = np.zeros((2, 4), dtype=np.float32)
    shared = InitialCollisionResult(
        robot_id="k1",
        fps=30.0,
        seconds=seconds,
        qpos=qpos,
        diagnostics=_initial_diagnostics(),
    )
    observed: dict[str, object] = {}

    def fake_shared_stage(
        forwarded_qpos: object,
        forwarded_seconds: object,
        *,
        robot_id: str,
        fps: float,
        progress: object,
        backend: object,
    ) -> InitialCollisionResult:
        observed.update(
            qpos=forwarded_qpos,
            seconds=forwarded_seconds,
            robot_id=robot_id,
            fps=fps,
            progress=progress,
            backend=backend,
        )
        return shared

    progress = cast(final_stage.FinalCollisionProgress, lambda *_: None)
    monkeypatch.setattr(final_stage, "_validate_reference_environment", lambda: None)
    monkeypatch.setattr(final_stage, "run_initial_collision_stage", fake_shared_stage)

    result = final_stage.run_final_collision(
        qpos,
        seconds,
        robot_id="K1",
        fps=30.0,
        contact_labels=labels,
        contact_confidence=confidence,
        progress=progress,
    )

    assert result.robot_id == "k1"
    assert observed["robot_id"] == "k1"
    assert observed["fps"] == 30.0
    assert observed["progress"] is progress
    assert observed["backend"] == "python"
    np.testing.assert_array_equal(observed["qpos"], qpos)
    np.testing.assert_array_equal(observed["seconds"], seconds)


@pytest.mark.parametrize(
    ("qpos", "seconds", "labels", "confidence", "message"),
    (
        (np.zeros((2, 35)), np.asarray([0.0, 0.1]), None, None, "shape"),
        (np.empty((0, 36)), np.empty(0), None, None, "at least one frame"),
        (
            np.zeros((2, 36)),
            np.asarray([0.0, 0.1]),
            np.zeros((2, 4)),
            None,
            "provided together",
        ),
        (
            np.zeros((2, 36)),
            np.asarray([0.0, 0.1]),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            "shape",
        ),
        (
            np.zeros((2, 36)),
            np.asarray([0.0, 0.1]),
            np.zeros((2, 4)),
            np.full((2, 4), 2.0),
            r"\[0, 1\]",
        ),
    ),
)
def test_run_final_collision_rejects_invalid_stage7_inputs_before_shared_solve(
    monkeypatch: pytest.MonkeyPatch,
    qpos: np.ndarray,
    seconds: np.ndarray,
    labels: np.ndarray | None,
    confidence: np.ndarray | None,
    message: str,
) -> None:
    called = False

    def unexpected(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal called
        del args, kwargs
        called = True
        return SimpleNamespace()

    monkeypatch.setattr(final_stage, "_validate_reference_environment", lambda: None)
    monkeypatch.setattr(final_stage, "run_initial_collision_stage", unexpected)

    with pytest.raises(MotionValidationError, match=message):
        final_stage.run_final_collision(
            qpos,
            seconds,
            robot_id="g1",
            fps=30.0,
            contact_labels=labels,
            contact_confidence=confidence,
        )
    assert called is False


def test_final_collision_rejects_nonreference_mujoco_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(final_stage.mujoco, "__version__", "3.5.0")

    with pytest.raises(ConfigurationError, match=r"MuJoCo 3\.6\.0.*3\.5\.0"):
        final_stage.run_final_collision(
            np.zeros((1, 36)),
            np.asarray([0.0]),
            robot_id="g1",
            fps=30.0,
        )
