from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import rimkit.pipeline.runner as runner
from rimkit.exceptions import ArtifactError, MotionValidationError
from rimkit.native import BackendSelection
from rimkit.optimization.fpa import FpaSolveRecord
from rimkit.pipeline.events import (
    CallbackEventSink,
    PipelineEvent,
    PipelineEventType,
)
from rimkit.pipeline.state import PipelineStage
from rimkit.stages.fpa import FPA_TARGET_SOLVE_LABELS


@dataclass(frozen=True)
class _Diagnostics:
    backend: str = "test"
    output_violations: int = 0


@dataclass(frozen=True)
class _Camera:
    name: str = "test"


class _Stage:
    def __init__(
        self,
        key: str,
        *,
        qpos: np.ndarray[Any, Any] | None = None,
    ) -> None:
        self.key = key
        self.fps = 30.0
        self.seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
        self.qpos = np.zeros((2, 36), dtype=np.float64) if qpos is None else qpos
        self.diagnostics = _Diagnostics()
        self.right_ground_distance_post = np.zeros(2, dtype=np.float64)
        self.left_ground_distance_post = np.zeros(2, dtype=np.float64)
        self.right_contact_weight = np.zeros(2, dtype=np.float64)
        self.left_contact_weight = np.zeros(2, dtype=np.float64)

    def reference_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            self.key: self.qpos,
        }


class _Contacts:
    fps = 30.0
    seconds = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
    right_confidence = np.ones(2, dtype=np.float64)
    left_confidence = np.ones(2, dtype=np.float64)

    def reference_arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "fps": np.asarray(self.fps, dtype=np.float64),
            "seconds": self.seconds,
            "right_contact_label": np.ones(2, dtype=np.bool_),
            "left_contact_label": np.ones(2, dtype=np.bool_),
        }


class _PreviewContacts:
    labels = np.asarray([[True, True, False, False]] * 2, dtype=np.bool_)
    confidence = np.asarray([[1.0, 1.0, 0.0, 0.0]] * 2, dtype=np.float64)
    availability = np.asarray([True, True, False, False], dtype=np.bool_)
    flight = np.zeros(2, dtype=np.bool_)
    contact_source = "test_feet"
    hand_contact_source = "unavailable_test"


def _target_solve_records() -> tuple[FpaSolveRecord, ...]:
    return tuple(
        FpaSolveRecord(label, "CLARABEL", "optimal", float(index))
        for index, label in enumerate(FPA_TARGET_SOLVE_LABELS)
    )


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
) -> tuple[_Stage, _Stage]:
    motion = SimpleNamespace(
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        frame_count=2,
        fps=30.0,
        duration_seconds=2.0 / 30.0,
    )
    contacts = _Contacts()
    dmr = _Stage("qpos_dmr_array", qpos=np.zeros((2, 36), dtype=np.float32))
    collision = _Stage("qpos_cc_smt_array")
    targets = _Stage("p_root_trgt_array")
    ara = _Stage("p_root_trgt_ara_array")
    ara.diagnostics = _Diagnostics()
    fpa_targets = _Stage("qpos_ara_array")
    fpa_targets.solve_records = _target_solve_records()
    fpa_ik = _Stage("qpos_fpa_array")
    fpa_ik.solve_records = ()
    fpa_ik.right_ground_distance_post[:] = (0.001, 0.002)
    fpa_ik.left_ground_distance_post[:] = (0.001, 0.002)
    fpa_ik.right_contact_weight[:] = 1.0
    fpa_ik.left_contact_weight[:] = 1.0
    final = _Stage("qpos_cc_fpa_array")
    diagnostics = _Stage("p_rt_cc_actual_array")

    monkeypatch.setattr(runner, "load_soma_motion", lambda *_args, **_kwargs: motion)
    monkeypatch.setattr(runner, "verify_robot", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    monkeypatch.setattr(runner, "build_contact_schedule", lambda _motion: contacts)
    monkeypatch.setattr(runner, "run_dmr", lambda *_args, **_kwargs: dmr)
    monkeypatch.setattr(runner, "run_initial_collision", lambda *_args, **_kwargs: collision)
    monkeypatch.setattr(runner, "run_target_trajectories", lambda *_args, **_kwargs: targets)
    monkeypatch.setattr(runner, "run_ara", lambda *_args, **_kwargs: ara)
    monkeypatch.setattr(runner, "build_fpa_targets", lambda *_args, **_kwargs: fpa_targets)
    monkeypatch.setattr(runner, "solve_fpa", lambda *_args, **_kwargs: fpa_ik)
    monkeypatch.setattr(
        runner,
        "build_preview_contact_state",
        lambda *_args, **_kwargs: _PreviewContacts(),
    )
    monkeypatch.setattr(runner, "run_final_collision", lambda *_args, **_kwargs: final)
    monkeypatch.setattr(
        runner,
        "run_diagnostic_trajectories",
        lambda *_args, **_kwargs: diagnostics,
    )

    def fake_export(path: Path, **_kwargs: object) -> SimpleNamespace:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, qpos=final.qpos)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(runner, "write_robot_motion_npz", fake_export)

    def fake_render(**kwargs: object) -> SimpleNamespace:
        video_path = kwargs["video_path"]
        thumbnail_path = kwargs["thumbnail_path"]
        assert video_path is None or isinstance(video_path, Path)
        assert thumbnail_path is None or isinstance(thumbnail_path, Path)
        width = kwargs["width"]
        height = kwargs["height"]
        assert isinstance(width, int)
        assert isinstance(height, int)
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"test-video")
        if thumbnail_path is not None:
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            thumbnail_path.write_bytes(b"test-thumbnail")
        return SimpleNamespace(
            video_path=video_path,
            thumbnail_path=thumbnail_path,
            width=width,
            height=height,
            fps=30.0,
            frame_count=2,
            camera=_Camera(),
            visualization_style="test",
            contact_overlay="test",
        )

    monkeypatch.setattr(runner, "render_motion_preview", fake_render)
    return fpa_targets, fpa_ik


def test_pipeline_runner_publishes_complete_safe_bundle_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)

    result = runner.run_retarget_pipeline(source, "g1", destination)

    assert result.output_dir == destination.resolve()
    assert result.output_dir.stat().st_mode & 0o777 == 0o755
    assert result.final_motion_path == destination / "final/robot_motion.npz"
    assert result.final_motion_path.is_file()
    assert result.video_path is None
    assert result.thumbnail_path is None
    assert tuple(result.stage_paths) == (
        "1_contacts",
        "2_dmr",
        "3_initial_collision",
        "4_target_trajectories",
        "5_ara",
        "6_fpa_targets",
        "7_fpa_ik",
        "8_final",
        "9_diagnostics",
    )
    for path in result.stage_paths.values():
        assert path.is_file()
        with np.load(path, allow_pickle=False) as archive:
            assert all(not archive[name].dtype.hasobject for name in archive.files)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["classification"] == "core-final-candidate"
    assert manifest["review_status"] == "unreviewed"
    assert manifest["pipeline_complete"] is True
    assert manifest["robot"]["id"] == "g1"
    assert manifest["source_motion"]["sha256"] == hashlib.sha256(b"safe-source").hexdigest()
    assert set(manifest["artifacts"]["stages"]) == set(result.stage_paths)
    assert manifest["diagnostics"]["ground"]["right_support"]["min_m"] == 0.001
    fpa_diagnostics = manifest["diagnostics"]["fpa"]
    assert fpa_diagnostics["expected_counts"] == {"stage_60": 8, "stage_70": 0}
    assert fpa_diagnostics["actual_counts"] == {"stage_60": 8, "stage_70": 0}
    assert fpa_diagnostics["records_complete"] is True
    assert fpa_diagnostics["solver_path_qualified"] is True
    assert [record["label"] for record in fpa_diagnostics["stage_60"]] == list(
        FPA_TARGET_SOLVE_LABELS
    )


def test_pipeline_runner_warns_for_fpa_fallback_and_inaccurate_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    fpa_targets, _ = _install_fake_pipeline(monkeypatch, source)
    records = list(fpa_targets.solve_records)
    records[0] = FpaSolveRecord(
        records[0].label,
        "OSQP",
        "optimal_inaccurate",
        records[0].objective,
    )
    fpa_targets.solve_records = tuple(records)

    result = runner.run_retarget_pipeline(source, "g1", destination)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    diagnostics = manifest["diagnostics"]["fpa"]
    assert diagnostics["records_complete"] is True
    assert diagnostics["all_reference_solver"] is False
    assert diagnostics["all_optimal"] is False
    assert diagnostics["solver_path_qualified"] is False
    assert sum("solver other than CLARABEL" in item for item in manifest["warnings"]) == 1
    assert sum("status other than optimal" in item for item in manifest["warnings"]) == 1


def test_pipeline_runner_refuses_existing_output_before_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    destination.mkdir()
    called = False

    def unexpected(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(runner, "load_soma_motion", unexpected)
    with pytest.raises(ArtifactError, match="already exists"):
        runner.run_retarget_pipeline(source, "g1", destination)
    assert not called


def test_pipeline_runner_rejects_one_frame_for_every_robot_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    monkeypatch.setattr(
        runner,
        "load_soma_motion",
        lambda *_args, **_kwargs: SimpleNamespace(frame_count=1),
    )
    model_loaded = False

    def unexpected_model_load(*_args: object, **_kwargs: object) -> object:
        nonlocal model_loaded
        model_loaded = True
        return object()

    monkeypatch.setattr(runner, "verify_robot", unexpected_model_load)

    with pytest.raises(MotionValidationError, match="complete CoRe pipeline.*two"):
        runner.run_retarget_pipeline(source, "k1", destination)

    assert not model_loaded
    assert not destination.exists()


def test_pipeline_runner_emits_ordered_stage_lifecycle_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)
    events: list[PipelineEvent] = []

    runner.run_retarget_pipeline(
        source,
        "g1",
        destination,
        render_video=True,
        render_thumbnail=True,
        event_sink=CallbackEventSink(events.append),
    )

    expected_stages = [
        PipelineStage.VALIDATING,
        PipelineStage.DMR,
        PipelineStage.INITIAL_COLLISION,
        PipelineStage.EXTRACTING_TRAJECTORIES,
        PipelineStage.ARA,
        PipelineStage.FPA_TARGETS,
        PipelineStage.FPA_IK,
        PipelineStage.FINAL_COLLISION,
        PipelineStage.VALIDATING_OUTPUT,
        PipelineStage.RENDERING,
        PipelineStage.EXPORTING,
    ]
    assert [
        event.stage for event in events if event.event_type == PipelineEventType.STAGE_STARTED
    ] == expected_stages
    assert [
        event.stage for event in events if event.event_type == PipelineEventType.STAGE_COMPLETED
    ] == expected_stages
    assert events[-1].event_type == PipelineEventType.COMPLETED
    assert events[-1].stage == PipelineStage.SUCCEEDED_WITH_WARNINGS
    assert len({event.job_id for event in events}) == 1
    assert {event.robot_id for event in events} == {"g1"}


def test_pipeline_runner_reports_sanitized_auto_fallback_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)
    monkeypatch.setattr(
        runner,
        "resolve_backend",
        lambda _backend: BackendSelection(
            requested="auto",
            selected="python",
            reason="native_extension_unavailable",
            module_name="rimkit._core_native",
            detail="ImportError: /private/build/path/libmujoco.dylib was not found",
        ),
    )
    events: list[PipelineEvent] = []

    result = runner.run_retarget_pipeline(
        source,
        "g1",
        destination,
        event_sink=CallbackEventSink(events.append),
    )

    warning = next(event for event in events if event.event_type == PipelineEventType.WARNING)
    assert warning.stage == PipelineStage.VALIDATING
    assert "Python reference backend" in warning.message
    assert "/private/build/path" not in warning.message
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["compute_backend"] == {
        "requested": "auto",
        "selected": "python",
        "reason": "native_extension_unavailable",
        "module": "rimkit._core_native",
    }
    assert any("Python reference backend" in item for item in manifest["warnings"])


def test_pipeline_runner_emits_failed_for_active_stage_and_cleans_temp_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)
    monkeypatch.setattr(
        runner,
        "run_ara",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ARA exploded")),
    )
    events: list[PipelineEvent] = []

    with pytest.raises(RuntimeError, match="ARA exploded"):
        runner.run_retarget_pipeline(
            source,
            "g1",
            destination,
            event_sink=CallbackEventSink(events.append),
        )

    assert events[-1].event_type == PipelineEventType.FAILED
    assert events[-1].stage == PipelineStage.ARA
    assert "RuntimeError: ARA exploded" in events[-1].message
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".result.building-*"))


def test_pipeline_runner_failed_sink_does_not_mask_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)

    def fail_ara(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("original pipeline failure")

    class _FailOnFailedSink:
        def emit(self, event: PipelineEvent) -> None:
            if event.event_type == PipelineEventType.FAILED:
                raise ValueError("event delivery failure")

    monkeypatch.setattr(runner, "run_ara", fail_ara)

    with pytest.raises(RuntimeError, match="original pipeline failure"):
        runner.run_retarget_pipeline(
            source,
            "g1",
            destination,
            event_sink=_FailOnFailedSink(),
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".result.building-*"))


def test_pipeline_runner_post_publish_sink_errors_do_not_invalidate_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"safe-source")
    destination = tmp_path / "result"
    _install_fake_pipeline(monkeypatch, source)
    events: list[PipelineEvent] = []

    class _FailOnPostPublishSink:
        def emit(self, event: PipelineEvent) -> None:
            events.append(event)
            if (
                event.event_type == PipelineEventType.STAGE_COMPLETED
                and event.stage == PipelineStage.EXPORTING
            ) or event.event_type == PipelineEventType.COMPLETED:
                raise RuntimeError("post-publication event delivery failed")

    result = runner.run_retarget_pipeline(
        source,
        "g1",
        destination,
        event_sink=_FailOnPostPublishSink(),
    )

    assert result.output_dir == destination.resolve()
    assert result.manifest_path.is_file()
    assert result.final_motion_path.is_file()
    export_completed = next(
        event
        for event in events
        if event.event_type == PipelineEventType.STAGE_COMPLETED
        and event.stage == PipelineStage.EXPORTING
    )
    assert export_completed.message == "Result bundle was atomically published."
    assert export_completed.metrics == {"artifact_count": 10.0}
    terminal = events[-1]
    assert terminal.event_type == PipelineEventType.COMPLETED
    assert terminal.stage == PipelineStage.SUCCEEDED_WITH_WARNINGS
    assert terminal.message == "Complete unreviewed candidate motion was published."
