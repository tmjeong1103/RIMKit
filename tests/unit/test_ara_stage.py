from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.motion.contacts import ContactSchedule
from core_retarget.stages import (
    AraDiagnostics,
    AraFloorStats,
    AraResult,
    AraSlipStats,
    TargetTrajectoriesResult,
    run_ara,
)

ARCHIVE_KEYS = (
    "schema_version",
    "fps",
    "seconds",
    "ground_offset",
    "s_val",
    "b_val",
    "p_root_trgt_grd_array",
    "p_ra_trgt_grd_array",
    "p_la_trgt_grd_array",
    "p_rt_trgt_grd_array",
    "p_lt_trgt_grd_array",
    "p_root_trgt_ara_array",
    "p_ra_trgt_ara_array",
    "p_la_trgt_ara_array",
    "p_rt_trgt_ara_array",
    "p_lt_trgt_ara_array",
    "root_xy_shift",
    "root_xy_expected_bound",
)


def _targets(*, robot_id: str = "g1", frame_count: int = 3) -> TargetTrajectoriesResult:
    seconds = np.arange(frame_count, dtype=np.float64) / 30.0
    root = np.column_stack(
        (
            np.linspace(0.0, 0.2, frame_count),
            np.linspace(0.0, 0.1, frame_count),
            np.full(frame_count, 0.8),
        )
    )
    right_ankle = root + np.asarray([0.0, -0.1, -0.65])
    left_ankle = root + np.asarray([0.0, 0.1, -0.65])
    right_toe = root + np.asarray([0.1, -0.1, -0.75])
    left_toe = root + np.asarray([0.1, 0.1, -0.75])
    return TargetTrajectoriesResult(
        robot_id=robot_id,
        fps=30.0,
        seconds=seconds,
        root=root,
        right_ankle=right_ankle,
        left_ankle=left_ankle,
        right_toe=right_toe,
        left_toe=left_toe,
        root_smoothed=root,
        right_ankle_smoothed=right_ankle,
        left_ankle_smoothed=left_ankle,
        right_toe_smoothed=right_toe,
        left_toe_smoothed=left_toe,
    )


def _contacts(
    targets: TargetTrajectoriesResult,
    *,
    seconds: np.ndarray | None = None,
    fps: float | None = None,
    right: np.ndarray | None = None,
    left: np.ndarray | None = None,
) -> ContactSchedule:
    return cast(
        ContactSchedule,
        SimpleNamespace(
            frame_count=targets.frame_count,
            seconds=targets.seconds if seconds is None else seconds,
            fps=targets.fps if fps is None else fps,
            right_contact_segments=(
                np.asarray([[0, targets.frame_count]], dtype=np.int64) if right is None else right
            ),
            left_contact_segments=(
                np.asarray([[0, targets.frame_count]], dtype=np.int64) if left is None else left
            ),
        ),
    )


def _diagnostics() -> AraDiagnostics:
    slip = AraSlipStats(0.0, 0.0, 0.0)
    floor = AraFloorStats(0.0, 0.0)
    return AraDiagnostics(
        backend="cvxpy",
        solver="CLARABEL",
        status="optimal",
        objective=0.0,
        lowest_contact_z=0.0,
        ground_translation_z=0.0,
        right_slip_pre=slip,
        right_slip_post=slip,
        left_slip_pre=slip,
        left_slip_post=slip,
        right_floor_z=floor,
        left_floor_z=floor,
        root_xy_bound_violation=-0.4,
    )


def test_ara_result_owns_read_only_arrays_and_exact_contract() -> None:
    targets = _targets()
    result = run_ara(targets, _contacts(targets), robot_id="g1")

    assert result.frame_count == targets.frame_count
    assert result.diagnostics.backend == "cvxpy"
    assert result.diagnostics.solver == "CLARABEL"
    assert result.diagnostics.status in ("optimal", "optimal_inaccurate")
    assert not result.root_ara.flags.writeable
    with pytest.raises(ValueError):
        result.root_ara[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.fps = 60.0  # type: ignore[misc]

    archive = result.reference_arrays()
    assert tuple(archive) == ARCHIVE_KEYS
    assert archive["schema_version"].dtype == np.dtype(np.int32)
    assert archive["fps"].dtype == np.dtype(np.float64)
    assert archive["p_root_trgt_ara_array"] is result.root_ara


def test_ara_global_ground_offset_uses_lowest_contact_segment_median() -> None:
    targets = _targets(frame_count=4)
    right_toe = targets.right_toe_smoothed.copy()
    left_toe = targets.left_toe_smoothed.copy()
    right_toe[:, 2] = (0.04, 0.06, 0.50, 0.50)
    left_toe[:, 2] = (0.20, 0.20, 0.10, 0.14)
    targets = TargetTrajectoriesResult(
        robot_id=targets.robot_id,
        fps=targets.fps,
        seconds=targets.seconds,
        root=targets.root,
        right_ankle=targets.right_ankle,
        left_ankle=targets.left_ankle,
        right_toe=right_toe,
        left_toe=left_toe,
        root_smoothed=targets.root_smoothed,
        right_ankle_smoothed=targets.right_ankle_smoothed,
        left_ankle_smoothed=targets.left_ankle_smoothed,
        right_toe_smoothed=right_toe,
        left_toe_smoothed=left_toe,
    )
    contacts = _contacts(
        targets,
        right=np.asarray([[0, 2]], dtype=np.int64),
        left=np.asarray([[2, 4]], dtype=np.int64),
    )

    result = run_ara(targets, contacts, robot_id="g1")

    assert result.diagnostics.lowest_contact_z == pytest.approx(0.05)
    np.testing.assert_array_equal(result.ground_offset, [[0.0, 0.0, -0.05]])
    np.testing.assert_array_equal(
        result.root_grounded,
        targets.root_smoothed + np.asarray([[0.0, 0.0, -0.05]]),
    )


def test_ara_without_contact_segments_grounds_the_global_lowest_toe() -> None:
    targets = _targets()
    empty = np.empty((0, 2), dtype=np.int64)

    result = run_ara(
        targets,
        _contacts(targets, right=empty, left=empty),
        robot_id="g1",
    )

    lowest = min(
        np.min(targets.right_toe_smoothed[:, 2]),
        np.min(targets.left_toe_smoothed[:, 2]),
    )
    assert result.ground_offset[0, 2] == pytest.approx(-lowest)
    np.testing.assert_allclose(result.scale, [[1.0, 1.0, 1.0]], atol=1e-7)
    np.testing.assert_allclose(result.shift, [[0.0, 0.0, 0.0]], atol=1e-7)


@pytest.mark.parametrize(
    ("contacts_factory", "message"),
    (
        (
            lambda target: _contacts(
                target,
                seconds=target.seconds + np.asarray([0.0, 0.0, 1e-6]),
            ),
            "timestamps",
        ),
        (lambda target: _contacts(target, fps=60.0), "same FPS"),
        (
            lambda target: _contacts(
                target,
                right=np.asarray([[0, target.frame_count + 1]], dtype=np.int64),
            ),
            "exceed",
        ),
    ),
)
def test_run_ara_validates_contact_contract(
    contacts_factory: object,
    message: str,
) -> None:
    targets = _targets()
    factory = cast(object, contacts_factory)
    contacts = factory(targets)  # type: ignore[operator]
    with pytest.raises(MotionValidationError, match=message):
        run_ara(targets, contacts, robot_id="g1")


def test_run_ara_rejects_robot_mismatch() -> None:
    targets = _targets(robot_id="h2")
    with pytest.raises(ConfigurationError, match="must match"):
        run_ara(targets, _contacts(targets), robot_id="g1")


def test_ara_result_validates_array_shapes() -> None:
    zeros = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="ground_offset"):
        AraResult(
            robot_id="g1",
            fps=30.0,
            seconds=np.asarray([0.0, 1.0 / 30.0]),
            ground_offset=np.zeros(3),
            scale=np.ones((1, 3)),
            shift=np.zeros((1, 3)),
            root_grounded=zeros,
            right_ankle_grounded=zeros,
            left_ankle_grounded=zeros,
            right_toe_grounded=zeros,
            left_toe_grounded=zeros,
            root_ara=zeros,
            right_ankle_ara=zeros,
            left_ankle_ara=zeros,
            right_toe_ara=zeros,
            left_toe_ara=zeros,
            root_xy_shift=np.zeros((2, 2)),
            root_xy_expected_bound=np.zeros((2, 2)),
            toe_floor_target_z=0.0,
            diagnostics=_diagnostics(),
        )
