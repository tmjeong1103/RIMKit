from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rimkit.pipeline.events import PipelineEvent, PipelineEventType
from rimkit.pipeline.runner import RetargetRunResult
from rimkit.pipeline.state import PipelineStage
from rimkit.web.jobs import (
    ArtifactNotFoundError,
    JobCapacityError,
    JobManager,
    JobNotFoundError,
)


def _fake_runner(
    motion_path: str | Path,
    robot_id: str,
    output_dir: str | Path,
    **kwargs: object,
) -> RetargetRunResult:
    source = Path(motion_path)
    destination = Path(output_dir)
    assert source.read_bytes() == b"motion"
    sink = kwargs["event_sink"]
    assert sink is not None
    sink.emit(
        PipelineEvent(
            event_type=PipelineEventType.PROGRESS,
            stage=PipelineStage.DMR,
            job_id="pipeline-id",
            robot_id=robot_id,
            message="Direct retargeting.",
            current=1,
            total=2,
        )
    )
    final_motion = destination / "final" / "robot_motion.npz"
    manifest = destination / "manifest.json"
    video = destination / "preview" / "final.mp4"
    thumbnail = destination / "preview" / "final.png"
    final_motion.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    final_motion.write_bytes(b"robot")
    manifest.write_text("{}\n", encoding="utf-8")
    video.write_bytes(b"video")
    thumbnail.write_bytes(b"image")
    return RetargetRunResult(
        robot_id=robot_id,
        output_dir=destination,
        manifest_path=manifest,
        final_motion_path=final_motion,
        stage_paths={},
        video_path=video,
        thumbnail_path=thumbnail,
    )


def test_job_manager_runs_pipeline_and_exposes_only_named_artifacts(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    job_id, input_path, output_dir = manager.allocate()
    input_path.write_bytes(b"motion")
    try:
        submitted = manager.submit(
            job_id=job_id,
            original_filename="walk.npz",
            input_path=input_path,
            output_dir=output_dir,
            robot_id="g1",
            render_video=True,
            save_stages=False,
            fps_override=None,
            width=1280,
            height=720,
        )
        assert submitted["status"] in {"queued", "running"}

        result = manager.wait_for_terminal(job_id, timeout=5)
        assert result["status"] == "succeeded"
        assert result["stage"] == "SUCCEEDED_WITH_WARNINGS"
        assert result["artifacts"] == ["manifest", "motion", "thumbnail", "video"]
        assert manager.artifact(job_id, "motion").read_bytes() == b"robot"
        with pytest.raises(ArtifactNotFoundError):
            manager.artifact(job_id, "../../source_motion")

        events = manager.events_after(job_id, 0)
        assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
        progress = next(event for event in events if event["event_type"] == "progress")
        assert progress["job_id"] == job_id
        assert progress["pipeline_job_id"] == "pipeline-id"
        assert progress["current"] == 1
    finally:
        manager.shutdown()


def test_discard_allocation_removes_only_unsubmitted_job_directory(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    job_id, input_path, _output_dir = manager.allocate()
    input_path.write_bytes(b"upload")
    job_root = input_path.parent.parent
    try:
        manager.discard_allocation(job_id)
        assert not job_root.exists()
    finally:
        manager.shutdown()


def test_job_manager_preserves_supported_source_suffixes(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    allocations: list[str] = []
    try:
        for suffix in (".npz", ".pt", ".PT"):
            job_id, input_path, _output_dir = manager.allocate(suffix)
            allocations.append(job_id)
            assert input_path.name == f"source_motion{suffix.lower()}"
            manager.discard_allocation(job_id)
            allocations.pop()

        with pytest.raises(ValueError, match="suffix"):
            manager.allocate(".txt")
    finally:
        for job_id in allocations:
            manager.discard_allocation(job_id)
        manager.shutdown()


def test_job_manager_caps_running_and_queued_work(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_runner(
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        **kwargs: object,
    ) -> RetargetRunResult:
        started.set()
        assert release.wait(timeout=5)
        return _fake_runner(motion_path, robot_id, output_dir, **kwargs)

    manager = JobManager(
        tmp_path / "runs",
        max_active_jobs=1,
        runner=blocking_runner,
    )
    job_id, input_path, output_dir = manager.allocate()
    input_path.write_bytes(b"motion")
    try:
        manager.submit(
            job_id=job_id,
            original_filename="walk.npz",
            input_path=input_path,
            output_dir=output_dir,
            robot_id="g1",
            render_video=True,
            save_stages=False,
            fps_override=None,
            width=1280,
            height=720,
        )
        assert started.wait(timeout=5)
        with pytest.raises(JobCapacityError, match="try again later"):
            manager.ensure_capacity()
    finally:
        release.set()
        manager.wait_for_terminal(job_id, timeout=5)
        manager.shutdown()


def test_job_manager_expires_terminal_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = [100.0]
    monkeypatch.setattr("rimkit.web.jobs.time.monotonic", lambda: clock[0])
    manager = JobManager(
        tmp_path / "runs",
        result_ttl_seconds=60.0,
        runner=_fake_runner,
    )
    job_id, input_path, output_dir = manager.allocate()
    input_path.write_bytes(b"motion")
    try:
        manager.submit(
            job_id=job_id,
            original_filename="walk.npz",
            input_path=input_path,
            output_dir=output_dir,
            robot_id="g1",
            render_video=True,
            save_stages=False,
            fps_override=None,
            width=1280,
            height=720,
        )
        manager.wait_for_terminal(job_id, timeout=5)
        job_root = input_path.parent.parent
        assert job_root.is_dir()

        clock[0] = 161.0
        with pytest.raises(JobNotFoundError):
            manager.get(job_id)
        assert not job_root.exists()
    finally:
        manager.shutdown()
