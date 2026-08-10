from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core_retarget.exceptions import ArtifactError
from core_retarget.native import BackendSelection
from core_retarget.render import PreviewContactState
from core_retarget.review import run_review
from core_retarget.robots.registry import get_robot
from core_retarget.stages import InitialCollisionDiagnostics


class _FakeDmrResult:
    robot_id = "g1"
    fps = 30.0
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
    qpos = np.zeros((2, 36), dtype=np.float32)

    def reference_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_dmr_array": self.qpos,
        }


class _FakeCollisionResult:
    robot_id = "g1"
    fps = 30.0
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
    qpos = np.zeros((2, 36), dtype=np.float64)
    diagnostics = InitialCollisionDiagnostics(
        backend="test",
        distance_backend="test",
        ik_backend="test",
        trajectory_backend="test",
        root_geom_count=1,
        collision_geom_count=2,
        raw_candidate_pair_count=3,
        candidate_pair_count=2,
        arm_joint_names=("left_shoulder_pitch_joint",),
        input_violations=1,
        input_max_frame_violations=1,
        output_violations=0,
        output_max_frame_violations=0,
        passes=(),
    )

    def reference_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "qpos_cc_smt_array": self.qpos,
        }


class _FakeRetargeter:
    def __init__(self, robot_id: str, config: object) -> None:
        del config
        self.robot = get_robot(robot_id)
        self.backend = BackendSelection(
            requested="python",
            selected="python",
            reason="test",
        )

    def preflight(self, source: Path) -> SimpleNamespace:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return SimpleNamespace(
            motion=SimpleNamespace(
                sha256=digest,
                frame_count=2,
                fps=30.0,
                duration_seconds=1.0 / 30.0,
            )
        )

    def run_dmr(self, source: Path, *, progress: object) -> _FakeDmrResult:
        del source, progress
        return _FakeDmrResult()

    def run_initial_collision(
        self,
        dmr: _FakeDmrResult,
        *,
        progress: object,
    ) -> _FakeCollisionResult:
        del dmr, progress
        return _FakeCollisionResult()


def _fake_contacts() -> PreviewContactState:
    return PreviewContactState(
        fps=30.0,
        seconds=np.asarray([0.0, 1.0 / 30.0]),
        labels=np.asarray([[True, True, False, False], [True, False, False, False]]),
        confidence=np.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        availability=np.asarray([True, True, False, False]),
        flight=np.asarray([False, False]),
        segment_ranges=np.asarray([[0, 2]]),
        segment_boundaries=np.asarray([0.0, 1.0]),
        contact_source="test",
        hand_contact_source="unavailable_test",
    )


def test_run_review_publishes_self_describing_stage_artifacts_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("core_retarget.review.Retargeter", _FakeRetargeter)
    monkeypatch.setattr(
        "core_retarget.review.load_soma_motion",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "core_retarget.review.build_preview_contact_state",
        lambda *_args, **_kwargs: _fake_contacts(),
    )
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    output = tmp_path / "review"

    result = run_review(source, "g1", output)

    assert result.output_dir == output.resolve()
    assert result.output_dir.stat().st_mode & 0o777 == 0o755
    assert result.video_path is None
    assert result.thumbnail_path is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["classification"] == "stage3-review"
    assert manifest["review_status"] == "unreviewed"
    assert manifest["pipeline_complete"] is False
    assert manifest["last_completed_stage"] == "initial_collision"
    assert manifest["robot"]["id"] == "g1"
    assert manifest["source_motion"]["sha256"] == hashlib.sha256(b"safe-source").hexdigest()
    assert manifest["diagnostics"]["output_violations"] == 0
    assert manifest["schema_version"] == 2
    assert set(manifest["artifacts"]) == {"contacts", "dmr", "initial_collision"}

    with np.load(result.contacts_path, allow_pickle=False) as archive:
        assert archive["format"].item() == "core-preview-contacts-v1"
        assert archive["contact_label_names"].tolist() == [
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ]
        assert archive["contact_labels"].shape == (2, 4)
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    with np.load(result.dmr_path, allow_pickle=False) as archive:
        assert archive["format"].item() == "core-dmr-review-v1"
        assert archive["robot_id"].item() == "g1"
        assert archive["qpos_dmr_array"].shape == (2, 36)
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    with np.load(result.initial_collision_path, allow_pickle=False) as archive:
        assert archive["format"].item() == "core-initial-collision-review-v1"
        assert archive["qpos_cc_smt_array"].shape == (2, 36)
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_run_review_refuses_to_mix_with_an_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    output = tmp_path / "review"
    output.mkdir()

    with pytest.raises(ArtifactError, match="already exists"):
        run_review(source, "g1", output)
