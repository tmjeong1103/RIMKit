from __future__ import annotations

import numpy as np
import pytest

import core_retarget.stages.diagnostics as diagnostic_stage
from core_retarget.exceptions import MotionValidationError
from core_retarget.stages.diagnostics import DiagnosticTrajectoriesResult


def test_diagnostic_result_owns_arrays_and_has_exact_archive_contract() -> None:
    source = np.zeros((2, 3), dtype=np.float32)
    result = DiagnosticTrajectoriesResult(
        robot_id="g1",
        fps=30.0,
        seconds=np.asarray([0.0, 1.0 / 30.0], dtype=np.float32),
        right_toe_before_collision=source,
        left_toe_before_collision=source,
        right_toe_final=source,
        left_toe_final=source,
    )
    source[:] = 5.0

    assert result.right_toe_final.dtype == np.dtype(np.float64)
    assert not result.right_toe_final.flags.writeable
    np.testing.assert_array_equal(result.right_toe_final, 0.0)
    assert tuple(result.reference_arrays()) == (
        "schema_version",
        "fps",
        "seconds",
        "p_rt_fpa_actual_array",
        "p_lt_fpa_actual_array",
        "p_rt_cc_actual_array",
        "p_lt_cc_actual_array",
    )


@pytest.mark.parametrize(
    ("before", "final", "seconds", "fps", "message"),
    (
        (np.zeros((2, 35)), np.zeros((2, 36)), np.asarray([0.0, 0.1]), 30.0, "pre"),
        (np.zeros((2, 36)), np.zeros((2, 35)), np.asarray([0.0, 0.1]), 30.0, "final"),
        (np.empty((0, 36)), np.empty((0, 36)), np.empty(0), 30.0, "at least one"),
        (
            np.zeros((2, 36)),
            np.full((2, 36), np.nan),
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
def test_stage9_validates_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    before: np.ndarray,
    final: np.ndarray,
    seconds: np.ndarray,
    fps: float,
    message: str,
) -> None:
    monkeypatch.setattr(
        diagnostic_stage.MujocoModel,
        "from_robot",
        lambda robot_id: pytest.fail(f"loaded model for invalid input: {robot_id}"),
    )
    with pytest.raises(MotionValidationError, match=message):
        diagnostic_stage.run_diagnostic_trajectories(
            before,
            final,
            seconds,
            robot_id="g1",
            fps=fps,
        )
