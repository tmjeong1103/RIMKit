from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rimkit.cli.main import main
from rimkit.review import ReviewRunResult


def test_review_command_routes_to_stage3_review_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    motion = tmp_path / "walk.npz"
    motion.write_bytes(b"motion")
    observed: dict[str, object] = {}

    def fake_run_review(
        motion_path: Path,
        robot_id: str,
        output_dir: Path,
        **kwargs: object,
    ) -> ReviewRunResult:
        observed.update(
            motion_path=motion_path,
            robot_id=robot_id,
            output_dir=output_dir,
            kwargs=kwargs,
        )
        return ReviewRunResult(
            output_dir=output_dir,
            manifest_path=output_dir / "manifest.json",
            contacts_path=output_dir / "stages/1_contacts.npz",
            dmr_path=output_dir / "stages/2_dmr.npz",
            initial_collision_path=output_dir / "stages/3_initial_collision.npz",
            video_path=output_dir / "preview/stage3.mp4",
            thumbnail_path=output_dir / "preview/stage3.png",
        )

    monkeypatch.setattr("rimkit.cli.main.run_review", fake_run_review)

    exit_code = main(
        [
            "review",
            str(motion),
            "--robot",
            "g1",
            "--output",
            str(tmp_path / "runs"),
            "--video",
            "--thumbnail",
            "--json",
        ]
    )

    assert exit_code == 0
    assert observed["motion_path"] == motion
    assert observed["robot_id"] == "g1"
    assert observed["output_dir"] == tmp_path / "runs" / "walk" / "g1"
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["render_video"] is True
    assert kwargs["render_thumbnail"] is True
    assert kwargs["width"] == 1280
    assert kwargs["height"] == 720
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline_complete"] is False
    assert payload["review_status"] == "unreviewed"
    assert payload["contacts"].endswith("stages/1_contacts.npz")


def test_run_command_routes_to_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    motion = tmp_path / "walk.npz"
    motion.write_bytes(b"motion")
    observed: dict[str, object] = {}

    class FakeRetargeter:
        def __init__(self, robot_id: str, config: object) -> None:
            observed.update(robot_id=robot_id, config=config)

        def run(self, motion_path: Path, output_dir: Path, **kwargs: object) -> object:
            observed.update(
                motion_path=motion_path,
                output_dir=output_dir,
                kwargs=kwargs,
            )
            return SimpleNamespace(
                robot_id="h2",
                output_dir=output_dir,
                manifest_path=output_dir / "manifest.json",
                final_motion_path=output_dir / "final/robot_motion.npz",
                stage_paths={"8_final": output_dir / "stages/8_final.npz"},
                video_path=output_dir / "preview/final.mp4",
                thumbnail_path=output_dir / "preview/final.png",
            )

    monkeypatch.setattr("rimkit.cli.main.Retargeter", FakeRetargeter)
    exit_code = main(
        [
            "run",
            str(motion),
            "--robot",
            "h2",
            "--output",
            str(tmp_path / "runs"),
            "--video",
            "--thumbnail",
            "--no-stages",
            "--json",
        ]
    )

    assert exit_code == 0
    assert observed["robot_id"] == "h2"
    assert observed["motion_path"] == motion
    assert observed["output_dir"] == tmp_path / "runs" / "walk" / "h2"
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["save_stages"] is False
    assert kwargs["render_video"] is True
    assert kwargs["render_thumbnail"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline_complete"] is True
    assert payload["review_status"] == "unreviewed"
    assert payload["robot_id"] == "h2"
    assert payload["final_motion"].endswith("final/robot_motion.npz")


def test_serve_command_routes_to_local_web_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_serve(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr("rimkit.web.server.serve", fake_serve)
    exit_code = main(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--runs-dir",
            str(tmp_path / "web-runs"),
            "--max-upload-mb",
            "64",
            "--max-frames",
            "1800",
            "--max-active-jobs",
            "3",
            "--result-ttl-minutes",
            "30",
            "--max-video-width",
            "1280",
            "--max-video-height",
            "720",
            "--disable-stage-archives",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "host": "127.0.0.1",
        "port": 8765,
        "runs_dir": tmp_path / "web-runs",
        "max_upload_mb": 64,
        "max_frames": 1800,
        "max_active_jobs": 3,
        "result_ttl_minutes": 30,
        "max_video_width": 1280,
        "max_video_height": 720,
        "allow_stage_archives": False,
    }
