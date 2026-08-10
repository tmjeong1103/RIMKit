from __future__ import annotations

import os
import stat
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import core_retarget.render.video as video
from core_retarget.render.contact import PreviewContactState
from core_retarget.robots.registry import get_robot


def _qpos(*, frames: int = 3, robot_id: str = "k1") -> np.ndarray:
    trajectory = np.zeros((frames, get_robot(robot_id).expected_nq), dtype=np.float32)
    if frames:
        trajectory[:, 2] = 0.8
        trajectory[:, 3] = 1.0
    return trajectory


def _contacts(*, frames: int = 3, fps: float = 30.0) -> PreviewContactState:
    return PreviewContactState(
        fps=fps,
        seconds=np.arange(frames, dtype=np.float64) / fps,
        labels=np.zeros((frames, 4), dtype=bool),
        confidence=np.zeros((frames, 4), dtype=np.float64),
        availability=np.asarray([True, True, False, False]),
        flight=np.ones(frames, dtype=bool),
        segment_ranges=np.asarray([[0, frames]]),
        segment_boundaries=np.asarray([0.0, 1.0]),
        contact_source="test",
        hand_contact_source="unavailable-test",
    )


def test_render_motion_preview_validates_then_returns_typed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _qpos()
    video_path = tmp_path / "nested" / "preview.mp4"
    thumbnail_path = tmp_path / "nested" / "thumbnail.png"
    camera = video.PreviewCamera(
        tracking="root_xy_fixed_z",
        lookat_z=0.72,
        distance=2.5,
        azimuth=135.0,
        elevation=-12.0,
    )
    observed: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> video.PreviewCamera:
        observed.update(kwargs)
        assert kwargs["qpos"].dtype == np.dtype(np.float64)
        assert not np.shares_memory(kwargs["qpos"], source)
        Path(kwargs["video_path"]).parent.mkdir(parents=True)
        Path(kwargs["video_path"]).write_bytes(b"mp4")
        Path(kwargs["thumbnail_path"]).write_bytes(b"png")
        return camera

    monkeypatch.setattr(video, "_render_preview_files", fake_render)

    result = video.render_motion_preview(
        robot_id="k1",
        qpos=source,
        fps=30,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
    )

    assert result == video.PreviewArtifacts(
        video_path=video_path.resolve(),
        thumbnail_path=thumbnail_path.resolve(),
        width=1280,
        height=720,
        fps=30.0,
        frame_count=3,
        thumbnail_frame=1,
        camera=camera,
        visualization_style=None,
        contact_overlay=False,
    )
    assert observed["spec"].robot_id == "k1"
    assert observed["fps"] == 30.0
    assert observed["width"] == 1280
    assert observed["height"] == 720
    assert observed["motion_name"] is None
    assert observed["contacts"] is None


@pytest.mark.parametrize(
    ("qpos", "message"),
    (
        (np.zeros(30), "shape"),
        (np.zeros((2, 29)), "shape"),
        (np.zeros((0, 30)), "at least one frame"),
        (np.full((2, 30), np.nan), "finite"),
        (np.full((2, 30), "bad"), "real numeric dtype"),
    ),
)
def test_qpos_validation_happens_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qpos: np.ndarray,
    message: str,
) -> None:
    monkeypatch.setattr(
        video,
        "_render_preview_files",
        lambda **_: pytest.fail("invalid qpos reached the rendering backend"),
    )

    with pytest.raises(video.PreviewRenderError, match=message):
        video.render_motion_preview(
            robot_id="k1",
            qpos=qpos,
            fps=30.0,
            video_path=tmp_path / "preview.mp4",
            thumbnail_path=None,
        )


@pytest.mark.parametrize(
    ("fps", "width", "height", "message"),
    (
        (0.0, 640, 480, "fps"),
        (float("nan"), 640, 480, "fps"),
        (30.0, 0, 480, "width"),
        (30.0, True, 480, "width"),
        (30.0, 640, -1, "height"),
        (30.0, 641, 480, "even"),
        (30.0, 640, 481, "even"),
    ),
)
def test_numeric_and_h264_dimension_validation(
    tmp_path: Path,
    fps: float,
    width: int,
    height: int,
    message: str,
) -> None:
    with pytest.raises(video.PreviewRenderError, match=message):
        video.render_motion_preview(
            robot_id="k1",
            qpos=_qpos(),
            fps=fps,
            video_path=tmp_path / "preview.mp4",
            thumbnail_path=None,
            width=width,
            height=height,
        )


def test_output_path_contract_rejects_missing_wrong_and_existing_targets(
    tmp_path: Path,
) -> None:
    with pytest.raises(video.PreviewRenderError, match="must be provided"):
        video.render_motion_preview(
            robot_id="k1",
            qpos=_qpos(),
            fps=30.0,
            video_path=None,
            thumbnail_path=None,
        )
    with pytest.raises(video.PreviewRenderError, match=r"\.mp4 suffix"):
        video.render_motion_preview(
            robot_id="k1",
            qpos=_qpos(),
            fps=30.0,
            video_path=tmp_path / "preview.mov",
            thumbnail_path=None,
        )
    with pytest.raises(video.PreviewRenderError, match=r"\.png suffix"):
        video.render_motion_preview(
            robot_id="k1",
            qpos=_qpos(),
            fps=30.0,
            video_path=None,
            thumbnail_path=tmp_path / "thumbnail.jpg",
        )

    existing = tmp_path / "preview.mp4"
    existing.write_bytes(b"user-data")
    with pytest.raises(video.PreviewRenderError, match="already exists"):
        video.render_motion_preview(
            robot_id="k1",
            qpos=_qpos(),
            fps=30.0,
            video_path=existing,
            thumbnail_path=None,
        )
    assert existing.read_bytes() == b"user-data"


def test_contact_overlay_requires_matching_name_frames_and_fps(tmp_path: Path) -> None:
    base = {
        "robot_id": "k1",
        "qpos": _qpos(),
        "fps": 30.0,
        "video_path": None,
        "thumbnail_path": tmp_path / "thumbnail.png",
    }
    with pytest.raises(video.PreviewRenderError, match="supplied together"):
        video.render_motion_preview(**base, motion_name="motion")
    with pytest.raises(video.PreviewRenderError, match="frame count"):
        video.render_motion_preview(
            **base,
            motion_name="motion",
            contacts=_contacts(frames=2),
        )
    with pytest.raises(video.PreviewRenderError, match="FPS"):
        video.render_motion_preview(
            **base,
            motion_name="motion",
            contacts=_contacts(fps=60.0),
        )


def test_png_only_render_allows_odd_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = video.PreviewCamera(
        tracking="root_xy_fixed_z",
        lookat_z=0.72,
        distance=2.0,
        azimuth=135.0,
        elevation=-12.0,
    )
    monkeypatch.setattr(video, "_render_preview_files", lambda **_: camera)

    result = video.render_motion_preview(
        robot_id="k1",
        qpos=_qpos(frames=2),
        fps=30.0,
        video_path=None,
        thumbnail_path=tmp_path / "thumbnail.png",
        width=641,
        height=481,
    )

    assert result.video_path is None
    assert result.width == 641
    assert result.height == 481
    assert result.thumbnail_frame == 1


def test_imageio_and_ffmpeg_are_loaded_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_imageio(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(video.importlib, "import_module", missing_imageio)
    with pytest.raises(video.PreviewRenderError, match="requires imageio"):
        video._load_imageio(require_ffmpeg=False)

    fake_imageio = ModuleType("imageio.v2")

    def missing_ffmpeg(name: str) -> ModuleType:
        if name == "imageio.v2":
            return fake_imageio
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(video.importlib, "import_module", missing_ffmpeg)
    with pytest.raises(video.PreviewRenderError, match="imageio-ffmpeg"):
        video._load_imageio(require_ffmpeg=True)


@pytest.mark.parametrize(
    ("robot_id", "distance", "lookat_z"),
    (
        ("g1", 2.65, 0.72),
        ("h1", 2.85, 0.82),
        ("h2", 3.05, 0.90),
        ("r1", 2.20, 0.55),
        ("k1", 2.65, 0.72),
        ("apollo", 3.00, 0.82),
        ("oli", 2.85, 0.82),
        ("n1", 2.55, 0.65),
        ("adam", 3.25, 0.92),
        ("t1", 2.45, 0.62),
        ("pm01", 2.45, 0.62),
    ),
)
def test_legacy_camera_presets_are_robot_specific(
    robot_id: str,
    distance: float,
    lookat_z: float,
) -> None:
    camera = video.preview_camera_for_robot(robot_id)

    assert camera.tracking == "root_xy_fixed_z"
    assert camera.distance == pytest.approx(distance)
    assert camera.lookat_z == pytest.approx(lookat_z)
    assert camera.azimuth == pytest.approx(135.0)
    assert camera.elevation == pytest.approx(-12.0)


def test_publish_temporaries_is_group_atomic_and_collaborator_readable(
    tmp_path: Path,
) -> None:
    temporary_video = tmp_path / ".preview.tmp.mp4"
    temporary_thumbnail = tmp_path / ".thumbnail.tmp.png"
    output_video = tmp_path / "preview.mp4"
    output_thumbnail = tmp_path / "thumbnail.png"
    temporary_video.write_bytes(b"video")
    temporary_thumbnail.write_bytes(b"image")
    temporary_video.chmod(0o600)
    temporary_thumbnail.chmod(0o600)

    video._publish_temporaries(
        {
            temporary_video: output_video,
            temporary_thumbnail: output_thumbnail,
        }
    )

    assert output_video.read_bytes() == b"video"
    assert output_thumbnail.read_bytes() == b"image"
    assert stat.S_IMODE(output_video.stat().st_mode) == 0o644
    assert stat.S_IMODE(output_thumbnail.stat().st_mode) == 0o644


def test_publish_failure_removes_outputs_from_the_same_render_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_video = tmp_path / ".preview.tmp.mp4"
    temporary_thumbnail = tmp_path / ".thumbnail.tmp.png"
    output_video = tmp_path / "preview.mp4"
    output_thumbnail = tmp_path / "thumbnail.png"
    temporary_video.write_bytes(b"video")
    temporary_thumbnail.write_bytes(b"image")
    real_replace = os.replace

    def fail_second(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == output_thumbnail:
            raise OSError("simulated second publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(video.os, "replace", fail_second)

    with pytest.raises(OSError, match="simulated"):
        video._publish_temporaries(
            {
                temporary_video: output_video,
                temporary_thumbnail: output_thumbnail,
            }
        )

    assert not output_video.exists()
    assert not output_thumbnail.exists()


def test_empty_temporary_is_never_published(tmp_path: Path) -> None:
    temporary = tmp_path / ".empty.mp4"
    output = tmp_path / "preview.mp4"
    temporary.touch()

    with pytest.raises(video.PreviewRenderError, match="empty preview"):
        video._publish_temporaries({temporary: output})

    assert not output.exists()


def test_render_failure_message_includes_backend_type_and_message() -> None:
    message = video._render_failure_message(
        robot_id="g1",
        error=RuntimeError("display backend failed"),
    )

    assert message == "Could not render the g1 preview: RuntimeError: display backend failed"
    assert "WindowServer" not in message


def test_invalid_coregraphics_failure_has_actionable_macos_guidance() -> None:
    message = video._render_failure_message(
        robot_id="h2",
        error=RuntimeError("invalid CoreGraphics connection"),
    )

    assert "RuntimeError: invalid CoreGraphics connection" in message
    assert "normal logged-in terminal session" in message
    assert "WindowServer access" in message
    assert "headless, SSH, or daemon session" in message


def test_overlay_setup_failure_is_wrapped_as_preview_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = SimpleNamespace(
        vis=SimpleNamespace(global_=SimpleNamespace(offwidth=0, offheight=0))
    )
    monkeypatch.setattr(video, "_load_imageio", lambda **_kwargs: ModuleType("imageio"))
    monkeypatch.setattr(video, "_load_scene", lambda _spec: (object(), fake_model, object()))
    monkeypatch.setattr(
        video,
        "load_overlay_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("Pillow missing")),
    )

    with pytest.raises(
        video.PreviewRenderError,
        match=r"Could not render the k1 preview: ModuleNotFoundError: Pillow missing",
    ):
        video._render_preview_files(
            spec=get_robot("k1"),
            qpos=np.asarray(_qpos(), dtype=np.float64),
            fps=30.0,
            motion_name="motion",
            contacts=_contacts(),
            video_path=None,
            thumbnail_path=tmp_path / "preview.png",
            width=1280,
            height=720,
        )
