from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from core_retarget import Retargeter
from core_retarget.exceptions import ConfigurationError
from core_retarget.stages import (
    DmrResult,
    InitialCollisionResult,
    TargetTrajectoriesResult,
)


def _stage_results(
    robot_id: str = "g1",
) -> tuple[DmrResult, InitialCollisionResult, np.ndarray, np.ndarray]:
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
    qpos_dmr = np.zeros((2, 36), dtype=np.float32)
    qpos_collision = np.zeros((2, 36), dtype=np.float64)
    dmr = cast(
        DmrResult,
        SimpleNamespace(
            robot_id=robot_id,
            fps=30.0,
            seconds=seconds,
            qpos=qpos_dmr,
        ),
    )
    collision = cast(
        InitialCollisionResult,
        SimpleNamespace(
            robot_id=robot_id,
            fps=30.0,
            seconds=seconds.copy(),
            qpos=qpos_collision,
        ),
    )
    return dmr, collision, qpos_dmr, qpos_collision


def test_retargeter_forwards_explicit_stage4_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dmr, collision, qpos_dmr, qpos_collision = _stage_results()
    sentinel = cast(TargetTrajectoriesResult, object())
    observed: dict[str, object] = {}

    def fake_stage(
        raw: object,
        smoothed: object,
        seconds: object,
        *,
        robot_id: str,
        fps: float,
    ) -> TargetTrajectoriesResult:
        observed.update(
            raw=raw,
            smoothed=smoothed,
            seconds=seconds,
            robot_id=robot_id,
            fps=fps,
        )
        return sentinel

    monkeypatch.setattr("core_retarget.api.run_target_trajectories_stage", fake_stage)

    result = Retargeter("g1").run_target_trajectories(dmr, collision)

    assert result is sentinel
    assert observed["raw"] is qpos_dmr
    assert observed["smoothed"] is qpos_collision
    assert observed["seconds"] is dmr.seconds
    assert observed["robot_id"] == "g1"
    assert observed["fps"] == 30.0


@pytest.mark.parametrize(("boundary", "robot_id"), (("DMR", "h1"), ("collision", "h1")))
def test_retargeter_rejects_cross_robot_stage4_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    robot_id: str,
) -> None:
    dmr, collision, _, _ = _stage_results()
    if boundary == "DMR":
        dmr.robot_id = robot_id  # type: ignore[misc]
    else:
        collision.robot_id = robot_id  # type: ignore[misc]
    called = False

    def unexpected(*args: Any, **kwargs: Any) -> TargetTrajectoriesResult:
        nonlocal called
        del args, kwargs
        called = True
        return cast(TargetTrajectoriesResult, object())

    monkeypatch.setattr("core_retarget.api.run_target_trajectories_stage", unexpected)
    with pytest.raises(ConfigurationError, match=f"{boundary} result robot"):
        Retargeter("g1").run_target_trajectories(dmr, collision)
    assert not called


def test_retargeter_rejects_mismatched_stage4_timelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dmr, collision, _, _ = _stage_results()
    collision.seconds[1] = 0.05  # type: ignore[misc]
    called = False

    def unexpected(*args: Any, **kwargs: Any) -> TargetTrajectoriesResult:
        nonlocal called
        del args, kwargs
        called = True
        return cast(TargetTrajectoriesResult, object())

    monkeypatch.setattr("core_retarget.api.run_target_trajectories_stage", unexpected)
    with pytest.raises(ConfigurationError, match="share one timeline"):
        Retargeter("g1").run_target_trajectories(dmr, collision)
    assert not called
