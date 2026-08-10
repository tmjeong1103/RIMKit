from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from core_retarget import Retargeter, RunConfig
from core_retarget.exceptions import ConfigurationError
from core_retarget.native import BackendSelection
from core_retarget.stages import DmrResult, InitialCollisionResult


@pytest.mark.parametrize(
    ("requested_robot", "normalized_robot", "qpos_dim"),
    (
        ("K1", "k1", 30),
        ("G1", "g1", 36),
        ("H1", "h1", 27),
        ("H2", "h2", 38),
        ("R1", "r1", 36),
    ),
)
def test_retargeter_initial_collision_forwards_the_explicit_dmr_stage_boundary(
    monkeypatch: pytest.MonkeyPatch,
    requested_robot: str,
    normalized_robot: str,
    qpos_dim: int,
) -> None:
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
    qpos = np.zeros((2, qpos_dim), dtype=np.float32)
    dmr_result = cast(
        DmrResult,
        SimpleNamespace(
            robot_id=normalized_robot,
            fps=30.0,
            seconds=seconds,
            qpos=qpos,
        ),
    )
    sentinel = cast(InitialCollisionResult, object())
    observed: dict[str, object] = {}

    def progress(outer: int, current: int, total: int, margin: float) -> None:
        del outer, current, total, margin

    def fake_run_initial_collision(
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
        return sentinel

    monkeypatch.setattr(
        "core_retarget.api.run_initial_collision_stage",
        fake_run_initial_collision,
    )

    result = Retargeter(
        requested_robot,
        RunConfig(robot=normalized_robot, backend="python"),
    ).run_initial_collision(
        dmr_result,
        progress=progress,
    )

    assert result is sentinel
    assert observed["qpos"] is qpos
    assert observed["seconds"] is seconds
    assert observed["robot_id"] == normalized_robot
    assert observed["fps"] == 30.0
    assert observed["progress"] is progress
    assert cast(BackendSelection, observed["backend"]).selected == "python"


def test_retargeter_initial_collision_rejects_a_dmr_result_for_another_robot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_stage(*args: Any, **kwargs: Any) -> InitialCollisionResult:
        nonlocal called
        del args, kwargs
        called = True
        return cast(InitialCollisionResult, object())

    monkeypatch.setattr(
        "core_retarget.api.run_initial_collision_stage",
        unexpected_stage,
    )
    dmr_result = cast(
        DmrResult,
        SimpleNamespace(
            robot_id="h1",
            fps=30.0,
            seconds=np.zeros(1, dtype=np.float64),
            qpos=np.zeros((1, 27), dtype=np.float32),
        ),
    )

    with pytest.raises(ConfigurationError, match="DMR result robot and Retargeter robot"):
        Retargeter("g1").run_initial_collision(dmr_result)

    assert not called
