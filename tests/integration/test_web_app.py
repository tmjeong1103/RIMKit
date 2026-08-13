from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import core_retarget.web.app as web_app_module
from core_retarget.pipeline.events import PipelineEvent, PipelineEventType
from core_retarget.pipeline.runner import RetargetRunResult
from core_retarget.pipeline.state import PipelineStage
from core_retarget.web.app import WebConfig, create_app
from core_retarget.web.jobs import JobCapacityError, JobManager

EXAMPLE = Path(__file__).parents[2] / "examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz"
EXAMPLE_MOTIONS_DIR = Path(__file__).parents[2] / "examples/motions"


def _fake_runner(
    motion_path: str | Path,
    robot_id: str,
    output_dir: str | Path,
    **kwargs: object,
) -> RetargetRunResult:
    assert Path(motion_path).is_file()
    destination = Path(output_dir)
    sink = kwargs["event_sink"]
    assert sink is not None
    sink.emit(
        PipelineEvent(
            event_type=PipelineEventType.STAGE_STARTED,
            stage=PipelineStage.DMR,
            job_id="runner-job",
            robot_id=robot_id,
            message="Running DMR.",
        )
    )
    final_motion = destination / "final" / "robot_motion.npz"
    manifest = destination / "manifest.json"
    video = destination / "preview" / "final.mp4"
    thumbnail = destination / "preview" / "final.png"
    final_motion.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    final_motion.write_bytes(b"robot-motion")
    manifest.write_text('{"pipeline_complete": true}\n', encoding="utf-8")
    video.write_bytes(b"video")
    thumbnail.write_bytes(b"png")
    return RetargetRunResult(
        robot_id=robot_id,
        output_dir=destination,
        manifest_path=manifest,
        final_motion_path=final_motion,
        stage_paths={},
        video_path=video,
        thumbnail_path=thumbnail,
    )


def test_web_app_validates_submits_streams_and_downloads(tmp_path: Path) -> None:
    observed_runner_options: dict[str, object] = {}

    def recording_runner(
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        **kwargs: object,
    ) -> RetargetRunResult:
        observed_runner_options.update(kwargs)
        return _fake_runner(motion_path, robot_id, output_dir, **kwargs)

    manager = JobManager(tmp_path / "runs", runner=recording_runner)
    app = create_app(
        WebConfig(
            runs_dir=tmp_path / "runs",
            example_motions_dir=EXAMPLE_MOTIONS_DIR,
        ),
        manager=manager,
    )
    try:
        with TestClient(app) as client:
            health = client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.json()["limits"]["max_frames"] == 1_000_000
            assert health.json()["source_formats"] == ["npz", "pt"]

            page = client.get("/")
            assert page.status_code == 200
            assert "Run CoRe retargeting" in page.text
            assert "Choose another motion" in page.text
            assert 'accept=".npz,.pt"' in page.text
            assert (
                "SOMA human motion in <code>.npz</code> (Kimodo) or <code>.pt</code> (Gem-X) format"
            ) in page.text
            assert 'data-testid="replace-motion"' in page.text
            assert 'data-testid="example-picker"' in page.text
            assert 'data-testid="robot-select"' in page.text
            assert 'id="selected-robot-name"' in page.text
            assert 'data-testid="robot-grid"' not in page.text
            assert '<option value="854x480" selected>' in page.text
            assert "/static/images/rilab_logo.jpg" in page.text
            assert "https://sites.google.com/view/sungjoon-choi/home" in page.text

            examples = client.get("/api/motions/examples")
            assert examples.status_code == 200
            assert [example["filename"] for example in examples.json()["examples"]] == [
                "foot_walk_stop.npz",
                "scurry_walk.pt",
            ]
            kimodo_example = examples.json()["examples"][0]
            example_download = client.get(kimodo_example["url"])
            assert example_download.status_code == 200
            assert (
                example_download.content
                == (EXAMPLE_MOTIONS_DIR / "kimodo/soma_rp_v11/foot_walk_stop.npz").read_bytes()
            )
            assert (
                'filename="foot_walk_stop.npz"' in example_download.headers["content-disposition"]
            )
            missing_example = client.get("/api/motions/examples/not-an-example")
            assert missing_example.status_code == 404

            rilab_logo = client.get("/static/images/rilab_logo.jpg")
            assert rilab_logo.status_code == 200
            assert rilab_logo.headers["content-type"] == "image/jpeg"
            assert rilab_logo.content.startswith(b"\xff\xd8\xff")

            robots = client.get("/api/robots").json()["robots"]
            assert [robot["id"] for robot in robots] == [
                "g1",
                "h1",
                "h2",
                "r1",
                "k1",
                "apollo",
                "oli",
                "n1",
                "adam",
                "t1",
                "pm01",
            ]
            assert robots[5:] == [
                {"id": "apollo", "name": "Apollo", "manufacturer": "Apptronik", "dof": 32},
                {"id": "oli", "name": "Oli", "manufacturer": "LimX Dynamics", "dof": 31},
                {"id": "n1", "name": "N1", "manufacturer": "Fourier Intelligence", "dof": 23},
                {"id": "adam", "name": "ADAM Lite", "manufacturer": "PNDbotics", "dof": 25},
                {"id": "t1", "name": "T1", "manufacturer": "Booster Robotics", "dof": 23},
                {"id": "pm01", "name": "PM01", "manufacturer": "ENGINEAI", "dof": 24},
            ]

            source = EXAMPLE.read_bytes()
            validation = client.post(
                "/api/motions/validate",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
            )
            assert validation.status_code == 200
            assert validation.json()["valid"] is True
            assert validation.json()["frame_count"] > 1

            created = client.post(
                "/api/jobs",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
                data={"robot": "h2", "render_video": "true", "save_stages": "false"},
            )
            assert created.status_code == 202
            job_id = created.json()["job_id"]
            result = manager.wait_for_terminal(job_id, timeout=5)
            assert result["status"] == "succeeded"
            assert observed_runner_options["width"] == 854
            assert observed_runner_options["height"] == 480

            status = client.get(f"/api/jobs/{job_id}")
            assert status.status_code == 200
            artifacts = status.json()["artifacts"]
            assert set(artifacts) == {"manifest", "motion", "thumbnail", "video"}

            events = client.get(f"/api/jobs/{job_id}/events")
            assert events.status_code == 200
            assert "event: message" in events.text
            assert "job_succeeded" in events.text

            motion = client.get(artifacts["motion"]["url"])
            assert motion.status_code == 200
            assert motion.content == b"robot-motion"
            assert "attachment" in motion.headers["content-disposition"]

            video = client.get(artifacts["video"]["url"])
            assert video.status_code == 200
            assert video.headers["content-type"] == "video/mp4"
            assert "inline" in video.headers["content-disposition"]

            video_range = client.get(
                artifacts["video"]["url"],
                headers={"Range": "bytes=0-2"},
            )
            assert video_range.status_code == 206
            assert video_range.content == b"vid"
            assert video_range.headers["content-range"] == "bytes 0-2/5"
    finally:
        manager.shutdown()


def test_web_app_accepts_gemx_pt_and_preserves_upload_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_suffixes: list[str] = []

    @dataclass(frozen=True)
    class FakeSourceSummary:
        path: Path
        sha256: str
        frame_count: int
        fps: float
        duration_seconds: float
        keys: tuple[str, ...]
        contact_channels: int | None
        warnings: tuple[str, ...]
        container_format: str
        provider: str

    def fake_validate_source_motion(
        path: str | Path,
        **_kwargs: object,
    ) -> FakeSourceSummary:
        source = Path(path)
        assert source.suffix == ".pt"
        assert source.read_bytes() == b"gemx-motion"
        return FakeSourceSummary(
            path=source,
            sha256="a" * 64,
            frame_count=12,
            fps=30.0,
            duration_seconds=0.4,
            keys=("body_params_global", "net_outputs"),
            contact_channels=6,
            warnings=(),
            container_format="pt",
            provider="gem-x",
        )

    def suffix_recording_runner(
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        **kwargs: object,
    ) -> RetargetRunResult:
        observed_suffixes.append(Path(motion_path).suffix)
        return _fake_runner(motion_path, robot_id, output_dir, **kwargs)

    monkeypatch.setattr(
        web_app_module,
        "validate_source_motion",
        fake_validate_source_motion,
    )
    manager = JobManager(tmp_path / "runs", runner=suffix_recording_runner)
    app = create_app(WebConfig(runs_dir=tmp_path / "runs"), manager=manager)
    try:
        with TestClient(app) as client:
            validation = client.post(
                "/api/motions/validate",
                files={"motion": ("rapid_stepping.PT", b"gemx-motion", "application/octet-stream")},
            )
            assert validation.status_code == 200
            assert validation.json()["container_format"] == "pt"
            assert validation.json()["provider"] == "gem-x"
            assert validation.json()["contact_channels"] == 6

            created = client.post(
                "/api/jobs",
                files={"motion": ("rapid_stepping.PT", b"gemx-motion", "application/octet-stream")},
                data={"robot": "g1", "render_video": "true"},
            )
            assert created.status_code == 202
            job_id = created.json()["job_id"]
            result = manager.wait_for_terminal(job_id, timeout=5)
            assert result["status"] == "succeeded"
            assert result["source_filename"] == "rapid_stepping.PT"
            assert observed_suffixes == [".pt"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("robot_id", ("apollo", "oli", "n1", "adam", "t1", "pm01"))
def test_web_app_submits_each_new_robot(robot_id: str, tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    app = create_app(WebConfig(runs_dir=tmp_path / "runs"), manager=manager)
    try:
        with TestClient(app) as client:
            source = EXAMPLE.read_bytes()
            created = client.post(
                "/api/jobs",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
                data={"robot": robot_id, "render_video": "true"},
            )
            assert created.status_code == 202
            result = manager.wait_for_terminal(created.json()["job_id"], timeout=5)
            assert result["status"] == "succeeded"
            assert result["robot_id"] == robot_id
    finally:
        manager.shutdown()


def test_web_app_enforces_public_deployment_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    app = create_app(
        WebConfig(
            runs_dir=tmp_path / "runs",
            max_frames=1800,
            max_video_width=1280,
            max_video_height=720,
            allow_stage_archives=False,
        ),
        manager=manager,
    )
    try:
        with TestClient(app) as client:
            source = EXAMPLE.read_bytes()
            oversized_video = client.post(
                "/api/jobs",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
                data={"robot": "g1", "width": "1920", "height": "1080"},
            )
            assert oversized_video.status_code == 422

            stage_archives = client.post(
                "/api/jobs",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
                data={"robot": "g1", "save_stages": "true"},
            )
            assert stage_archives.status_code == 422

            def reject_capacity() -> None:
                raise JobCapacityError("public queue full")

            monkeypatch.setattr(manager, "ensure_capacity", reject_capacity)
            queue_full = client.post(
                "/api/jobs",
                files={"motion": ("walk.npz", source, "application/octet-stream")},
                data={"robot": "g1"},
            )
            assert queue_full.status_code == 429
            assert queue_full.headers["retry-after"] == "30"
    finally:
        manager.shutdown()


def test_web_app_rejects_unsupported_uploads_and_unknown_artifacts(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "runs", runner=_fake_runner)
    app = create_app(WebConfig(runs_dir=tmp_path / "runs", max_upload_bytes=16), manager=manager)
    try:
        with TestClient(app) as client:
            wrong_type = client.post(
                "/api/motions/validate",
                files={"motion": ("walk.txt", b"text", "text/plain")},
            )
            assert wrong_type.status_code == 415

            too_large = client.post(
                "/api/motions/validate",
                files={"motion": ("walk.npz", b"x" * 17, "application/octet-stream")},
            )
            assert too_large.status_code == 413

            missing = client.get("/api/jobs/00000000000000000000000000000000")
            assert missing.status_code == 404
    finally:
        manager.shutdown()
