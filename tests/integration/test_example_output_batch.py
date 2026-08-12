from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import generate_example_outputs as batch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_npz(path: Path, *, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        fps=np.asarray(30.0, dtype=np.float64),
        seconds=np.asarray([0.0, 1.0 / 30.0], dtype=np.float64),
        **{key: np.zeros((2, 3), dtype=np.float64)},
    )


def _fake_pipeline_runner(calls: list[tuple[str, str]]) -> Any:
    def run_pipeline(
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        *,
        fps_override: float | None,
        save_stages: bool,
        render_video: bool,
        render_thumbnail: bool,
        width: int,
        height: int,
    ) -> SimpleNamespace:
        motion = Path(motion_path)
        destination = Path(output_dir)
        assert save_stages is True
        assert render_video is True
        assert render_thumbnail is True
        assert (width, height) == (1280, 720)
        assert fps_override == (30.0 if motion.suffix == ".pt" else None)
        calls.append((motion.stem, robot_id))

        destination.mkdir(parents=True, exist_ok=False)
        stage_paths: dict[str, Path] = {}
        for index, stage_name in enumerate(batch.REQUIRED_STAGES):
            path = destination / "stages" / f"{stage_name}.npz"
            _write_npz(path, key=f"stage_{index}")
            stage_paths[stage_name] = path

        final_motion_path = destination / "final/robot_motion.npz"
        _write_npz(final_motion_path, key="qpos")
        video_path = destination / "preview/final.mp4"
        thumbnail_path = destination / "preview/final.png"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42-safe-final-preview")
        thumbnail_path.write_bytes(b"\x89PNG\r\n\x1a\n-safe-final-thumbnail")

        manifest = {
            "schema_version": batch.PIPELINE_MANIFEST_SCHEMA_VERSION,
            "classification": "core-final-candidate",
            "review_status": "unreviewed",
            "pipeline_complete": True,
            "source_motion": {"sha256": _sha256(motion)},
            "robot": {"id": robot_id},
            "artifacts": {
                "final_motion": _artifact_record(final_motion_path, destination),
                "stages": {
                    name: _artifact_record(path, destination) for name, path in stage_paths.items()
                },
                "preview": {
                    "style": "legacy-contact-overlay-v1",
                    "contact_overlay": True,
                    "trajectory_stage": "final",
                    "width": 1280,
                    "height": 720,
                    "video": _artifact_record(video_path, destination),
                    "thumbnail": _artifact_record(thumbnail_path, destination),
                },
            },
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            output_dir=destination,
            manifest_path=manifest_path,
            final_motion_path=final_motion_path,
            stage_paths=stage_paths,
            video_path=video_path,
            thumbnail_path=thumbnail_path,
        )

    return run_pipeline


def _selected_args(tmp_path: Path, *, gallery: bool) -> list[str]:
    args = [
        "--motion",
        "stand_walk_run_stop",
        "--robot",
        "g1",
        "--robot",
        "k1",
        "--output",
        str(tmp_path / "runs"),
    ]
    if gallery:
        args.extend(("--gallery-dir", str(tmp_path / "gallery")))
    return args


def test_selected_batch_generates_complete_gallery_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(batch, "_load_pipeline_runner", lambda: _fake_pipeline_runner(calls))
    args = _selected_args(tmp_path, gallery=True)

    assert batch.main(args) == 0
    assert calls == [
        ("stand_walk_run_stop", "g1"),
        ("stand_walk_run_stop", "k1"),
    ]

    gallery = tmp_path / "gallery"
    assert {
        path.relative_to(gallery).as_posix() for path in gallery.rglob("*") if path.is_file()
    } == {
        "index.json",
        "stand_walk_run_stop/g1.mp4",
        "stand_walk_run_stop/g1.png",
        "stand_walk_run_stop/k1.mp4",
        "stand_walk_run_stop/k1.png",
    }
    index = json.loads((gallery / "index.json").read_text(encoding="utf-8"))
    assert (gallery / "index.json").stat().st_mode & 0o777 == 0o644
    assert index["classification"] == "core-final-candidate-gallery"
    assert index["review_status"] == "unreviewed"
    assert index["pipeline_complete"] is True
    assert index["visualization"] == {
        "style": "legacy-contact-overlay-v1",
        "width": 1280,
        "height": 720,
        "contact_overlay": True,
        "trajectory_stage": "final",
    }
    assert [record["path"] for record in index["files"]] == [
        "stand_walk_run_stop/g1.mp4",
        "stand_walk_run_stop/g1.png",
        "stand_walk_run_stop/k1.mp4",
        "stand_walk_run_stop/k1.png",
    ]
    for record in index["files"]:
        artifact = gallery / record["path"]
        assert record["size_bytes"] == artifact.stat().st_size
        assert record["sha256"] == _sha256(artifact)

    for robot_id in ("g1", "k1"):
        output_dir = tmp_path / "runs/stand_walk_run_stop" / robot_id
        assert (output_dir / "manifest.json").is_file()
        assert (output_dir / "final/robot_motion.npz").is_file()
        assert (output_dir / "preview/final.mp4").is_file()
        assert (output_dir / "preview/final.png").is_file()
        assert (output_dir / batch.RESULT_RECORD_NAME).is_file()
        assert len(tuple((output_dir / "stages").glob("*.npz"))) == 9

    assert batch.main([*args, "--resume"]) == 0
    assert calls == [
        ("stand_walk_run_stop", "g1"),
        ("stand_walk_run_stop", "k1"),
    ]
    output = capsys.readouterr().out
    assert "RESUME verified complete result" in output
    assert "0 generated, 2 resumed, 2 total" in output


@pytest.mark.parametrize(
    "tampered_relative_path",
    (
        "stages/5_ara.npz",
        "final/robot_motion.npz",
        "preview/final.mp4",
        "manifest.json",
    ),
    ids=("stage", "final-motion", "media", "manifest"),
)
def test_resume_rejects_tampered_pipeline_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampered_relative_path: str,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(batch, "_load_pipeline_runner", lambda: _fake_pipeline_runner(calls))
    args = [
        "--motion",
        "march_in_place_contacts",
        "--robot",
        "h2",
        "--output",
        str(tmp_path / "runs"),
    ]
    assert batch.main(args) == 0
    assert calls == [("march_in_place_contacts", "h2")]

    artifact = tmp_path / "runs/march_in_place_contacts/h2" / tampered_relative_path
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(batch.BatchGenerationError, match="mismatch"):
        batch.main([*args, "--resume"])
    assert calls == [("march_in_place_contacts", "h2")]


def test_gallery_flag_without_path_uses_documented_default() -> None:
    args = batch.parse_args(["--gallery-dir"])

    assert args.gallery_dir == batch.DEFAULT_GALLERY
    assert args.source_set == "kimodo"


def test_gemx_source_set_runs_bundled_pt_at_30_hz(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(batch, "_load_pipeline_runner", lambda: _fake_pipeline_runner(calls))

    assert (
        batch.main(
            [
                "--source-set",
                "gem-x",
                "--motion",
                "rapid_stepping",
                "--robot",
                "g1",
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    assert calls == [("rapid_stepping", "g1")]


def test_source_set_rejects_motion_from_other_provider() -> None:
    with pytest.raises(batch.BatchGenerationError, match="do not belong to the gem-x"):
        batch.main(["--source-set", "gem-x", "--motion", "stand_walk_run_stop"])


def test_rerender_gallery_reuses_complete_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(batch, "_load_pipeline_runner", lambda: _fake_pipeline_runner(calls))
    args = _selected_args(tmp_path, gallery=True)
    assert batch.main(args) == 0

    rendered: list[tuple[str, str]] = []

    def fake_rerender(
        output: batch.GalleryRenderInput,
        gallery_dir: Path,
        *,
        motion_path: Path,
        fps_override: float | None,
    ) -> None:
        assert motion_path.suffix == ".npz"
        assert fps_override is None
        rendered.append((output.motion_id, output.robot_id))
        destination = gallery_dir / output.motion_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{output.robot_id}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42-refreshed")
        (destination / f"{output.robot_id}.png").write_bytes(b"\x89PNG\r\n\x1a\n-refreshed")

    monkeypatch.setattr(batch, "_rerender_gallery_output", fake_rerender)
    assert batch.main([*args, "--rerender-gallery"]) == 0
    assert rendered == [
        ("stand_walk_run_stop", "g1"),
        ("stand_walk_run_stop", "k1"),
    ]
    assert calls == [
        ("stand_walk_run_stop", "g1"),
        ("stand_walk_run_stop", "k1"),
    ]

    gallery = tmp_path / "gallery"
    index = json.loads((gallery / "index.json").read_text(encoding="utf-8"))
    assert len(index["files"]) == 4
    for record in index["files"]:
        artifact = gallery / record["path"]
        assert record["sha256"] == _sha256(artifact)


def test_rerender_gallery_forwards_gemx_camera_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final_motion = tmp_path / "final/robot_motion.npz"
    final_motion.parent.mkdir(parents=True)
    np.savez_compressed(
        final_motion,
        robot_id=np.asarray("g1"),
        fps=np.asarray(30.0),
        qpos=np.zeros((2, 3), dtype=np.float64),
    )
    motion = SimpleNamespace(frame_count=2, fps=30.0)
    core_package = ModuleType("core_retarget")
    core_package.__path__ = []  # type: ignore[attr-defined]
    motion_package = ModuleType("core_retarget.motion")
    motion_package.load_source_motion = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: SimpleNamespace(motion=motion)
    )
    render_package = ModuleType("core_retarget.render")
    render_package.build_preview_contact_state = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: object()
    )
    observed: dict[str, str] = {}

    def fake_render_motion_preview(**kwargs: Any) -> None:
        observed["source_provider"] = kwargs["source_provider"]
        kwargs["video_path"].write_bytes(b"\x00\x00\x00\x18ftypmp42-gemx")
        kwargs["thumbnail_path"].write_bytes(b"\x89PNG\r\n\x1a\n-gemx")

    render_package.render_motion_preview = fake_render_motion_preview  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core_retarget", core_package)
    monkeypatch.setitem(sys.modules, "core_retarget.motion", motion_package)
    monkeypatch.setitem(sys.modules, "core_retarget.render", render_package)

    batch._rerender_gallery_output(
        batch.GalleryRenderInput(
            motion_id="rapid_stepping",
            robot_id="g1",
            final_motion_path=final_motion,
        ),
        tmp_path / "gallery",
        motion_path=tmp_path / "rapid_stepping.pt",
        fps_override=30.0,
    )

    assert observed["source_provider"] == "gem-x"
