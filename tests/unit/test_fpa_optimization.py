from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

import rimkit.stages.fpa as fpa_stage
from rimkit.optimization.fpa import FpaSolveRecord, FpaTrajectoryResult
from rimkit.stages.fpa import FPA_IK_SOLVE_LABELS


def test_fpa_solve_record_is_immutable_normalized_and_strict_json_safe() -> None:
    record = FpaSolveRecord(
        label=" stage6.right_toe.remain.x ",
        solver="clarabel",
        status="OPTIMAL",
        objective=np.float64(1.25),
    )

    assert record.label == "stage6.right_toe.remain.x"
    assert record.solver == "CLARABEL"
    assert record.status == "optimal"
    assert record.objective == 1.25
    assert not hasattr(record, "__dict__")
    assert json.loads(json.dumps(asdict(record), allow_nan=False))["objective"] == 1.25
    with pytest.raises(FrozenInstanceError):
        record.solver = "OSQP"  # type: ignore[misc]


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), float("-inf")])
def test_fpa_solve_record_rejects_nonfinite_objective(objective: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        FpaSolveRecord("stage6.test", "CLARABEL", "optimal", objective)


@pytest.mark.parametrize(
    ("label", "solver", "status"),
    [
        ("", "CLARABEL", "optimal"),
        ("stage6.test", "", "optimal"),
        ("stage6.test", "CLARABEL", "failed"),
    ],
)
def test_fpa_solve_record_rejects_invalid_metadata(
    label: str,
    solver: str,
    status: str,
) -> None:
    with pytest.raises(ValueError):
        FpaSolveRecord(label, solver, status, 0.0)


class _SmoothingContacts:
    def __init__(self, frame_count: int) -> None:
        self.base_positions_smoothed = np.zeros((frame_count, 3), dtype=np.float64)
        self.flight_label = np.zeros(frame_count, dtype=np.bool_)


def test_stage7_smoothing_records_all_nine_solves_in_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_pinned(
        seconds: np.ndarray[tuple[int], np.dtype[np.float64]],
        reference: np.ndarray[tuple[int], np.dtype[np.float64]],
        *,
        jerk_weight: float,
    ) -> FpaTrajectoryResult:
        nonlocal call_count
        del seconds, jerk_weight
        call_count += 1
        return FpaTrajectoryResult(
            values=reference,
            status="optimal",
            objective=float(call_count),
            solver="CLARABEL",
        )

    monkeypatch.setattr(fpa_stage, "shape_pinned_trajectory", fake_pinned)
    frame_count = 4
    seconds = np.arange(frame_count, dtype=np.float64) / 30.0
    qpos_ara = np.zeros((frame_count, 3), dtype=np.float64)
    contact = np.ones(frame_count, dtype=np.float64)
    adaptive = fpa_stage._adaptive_ara_smoothing(
        qpos_ara,
        _SmoothingContacts(frame_count),  # type: ignore[arg-type]
        contact,
        contact,
        seconds,
        1.0 / 30.0,
    )
    base = fpa_stage._smooth_base_correction(
        qpos_ara,
        qpos_ara,
        qpos_ara,
        seconds,
    )
    records = (*adaptive[-1], *base[-1])

    assert call_count == 9
    assert tuple(record.label for record in records) == FPA_IK_SOLVE_LABELS
    assert tuple(record.objective for record in records) == tuple(
        float(index) for index in range(1, 10)
    )


def test_stage7_short_clip_skips_all_pinned_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> FpaTrajectoryResult:
        raise AssertionError("short-clip smoothing must not call a trajectory solver")

    monkeypatch.setattr(fpa_stage, "shape_pinned_trajectory", unexpected)
    frame_count = 2
    seconds = np.arange(frame_count, dtype=np.float64) / 30.0
    qpos_ara = np.zeros((frame_count, 3), dtype=np.float64)
    contact = np.ones(frame_count, dtype=np.float64)
    adaptive = fpa_stage._adaptive_ara_smoothing(
        qpos_ara,
        _SmoothingContacts(frame_count),  # type: ignore[arg-type]
        contact,
        contact,
        seconds,
        1.0 / 30.0,
    )
    base = fpa_stage._smooth_base_correction(
        qpos_ara,
        qpos_ara,
        qpos_ara,
        seconds,
    )

    assert adaptive[-1] == ()
    assert base[-1] == ()
