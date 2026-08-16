from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rimkit import Retargeter, RunConfig
from rimkit.exceptions import MotionValidationError
from rimkit.stages import DmrResult

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "motions" / "kimodo" / "soma_rp_v11" / "stand_walk_run_stop.npz"


@pytest.mark.parametrize(
    ("requested_robot", "normalized_robot"),
    (
        ("K1", "k1"),
        ("G1", "g1"),
        ("H1", "h1"),
        ("H2", "h2"),
        ("R1", "r1"),
        ("Apollo", "apollo"),
        ("Oli", "oli"),
        ("N1", "n1"),
        ("Adam", "adam"),
        ("T1", "t1"),
        ("PM01", "pm01"),
    ),
)
def test_retargeter_run_dmr_loads_motion_and_forwards_selected_robot(
    monkeypatch: Any,
    requested_robot: str,
    normalized_robot: str,
) -> None:
    observed: dict[str, object] = {}
    sentinel = cast(DmrResult, object())

    def fake_run_dmr(
        motion: Any,
        *,
        robot_id: str,
        progress: Any,
        backend: Any,
    ) -> DmrResult:
        observed.update(
            robot_id=robot_id,
            frame_count=motion.frame_count,
            fps=motion.fps,
            progress=progress,
            backend=backend.selected,
        )
        return sentinel

    def callback(current: int, total: int, error: float) -> None:
        del current, total, error

    monkeypatch.setattr("rimkit.api.run_dmr_stage", fake_run_dmr)

    result = Retargeter(
        requested_robot,
        RunConfig(robot=normalized_robot, backend="python"),
    ).run_dmr(EXAMPLE, progress=callback)

    assert result is sentinel
    assert observed == {
        "robot_id": normalized_robot,
        "frame_count": 150,
        "fps": 30.0,
        "progress": callback,
        "backend": "python",
    }


def test_retargeter_preflight_rejects_one_frame_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rimkit.api.validate_soma_npz",
        lambda *_args, **_kwargs: SimpleNamespace(frame_count=1),
    )
    model_loaded = False

    def unexpected_model_load(*_args: object, **_kwargs: object) -> object:
        nonlocal model_loaded
        model_loaded = True
        return object()

    monkeypatch.setattr("rimkit.api.verify_robot", unexpected_model_load)

    with pytest.raises(MotionValidationError, match="complete CoRe pipeline.*two"):
        Retargeter("k1").preflight("one-frame.npz")

    assert not model_loaded
