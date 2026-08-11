"""Headless MuJoCo previews for robot qpos trajectories.

The renderer deliberately accepts an already-retargeted qpos trajectory.  It
does not infer a pipeline stage or promote a motion to a final CoRe export.
Video dependencies are imported only when this API is called, so importing the
main package does not require an FFmpeg installation or initialize OpenGL.
"""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.assets import root_path
from core_retarget.exceptions import CoReError
from core_retarget.render.contact import PreviewContactState
from core_retarget.render.legacy_visualization import (
    LEGACY_HEIGHT,
    LEGACY_VISUALIZATION_STYLE,
    LEGACY_WIDTH,
    annotate_frame,
    append_contact_markers,
    forward_pose,
    load_overlay_runtime,
    prepare_contact_scene,
)
from core_retarget.robots.profiles import get_dmr_profile
from core_retarget.robots.registry import get_robot
from core_retarget.robots.schema import RobotSpec

FloatArray = NDArray[np.float64]
RenderProgress = Callable[[int, int], None]
SourceProvider = Literal["kimodo", "gem-x"]

_DEFAULT_AZIMUTH = 135.0
_DEFAULT_ELEVATION = -12.0
_CAMERA_PRESETS = {
    "g1": (2.65, 0.72),
    "h1": (2.85, 0.82),
    "h2": (3.05, 0.90),
    "r1": (2.20, 0.55),
    "k1": (2.65, 0.72),
    "apollo": (3.00, 0.82),
    "oli": (2.85, 0.82),
    "n1": (2.55, 0.65),
    "adam": (3.25, 0.92),
    "t1": (2.45, 0.62),
    "pm01": (2.45, 0.62),
}


class PreviewRenderError(CoReError):
    """Raised when a preview request or rendering backend is invalid."""


@dataclass(frozen=True, slots=True)
class PreviewCamera:
    """Stable free-camera parameters used for every output frame."""

    tracking: Literal["root_xy_fixed_z"]
    lookat_z: float
    distance: float
    azimuth: float
    elevation: float


@dataclass(frozen=True, slots=True)
class PreviewArtifacts:
    """Files and rendering metadata produced for one qpos trajectory."""

    video_path: Path | None
    thumbnail_path: Path | None
    width: int
    height: int
    fps: float
    frame_count: int
    thumbnail_frame: int | None
    camera: PreviewCamera
    visualization_style: str | None
    contact_overlay: bool


def _validated_qpos(qpos: ArrayLike, *, spec: RobotSpec) -> FloatArray:
    try:
        untyped = np.asarray(qpos)
    except (TypeError, ValueError) as error:
        raise PreviewRenderError("qpos must be a numeric two-dimensional array") from error
    if untyped.dtype.kind not in "fiu":
        raise PreviewRenderError("qpos must use a real numeric dtype")
    trajectory = np.asarray(untyped, dtype=np.float64)
    expected_shape = f"(frames, {spec.expected_nq})"
    if trajectory.ndim != 2 or trajectory.shape[1:] != (spec.expected_nq,):
        raise PreviewRenderError(f"qpos shape must be {expected_shape}; found {trajectory.shape}")
    if len(trajectory) == 0:
        raise PreviewRenderError("qpos must contain at least one frame")
    if not np.isfinite(trajectory).all():
        raise PreviewRenderError("qpos must contain only finite values")
    return trajectory.copy(order="C")


def _validated_output_path(path: Path, *, suffix: str, field: str) -> Path:
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise PreviewRenderError(f"{field} could not be resolved: {path}") from error
    if resolved.suffix != suffix:
        raise PreviewRenderError(f"{field} must use the {suffix} suffix")
    if resolved.exists():
        kind = "directory" if resolved.is_dir() else "file"
        raise PreviewRenderError(f"{field} already exists as a {kind}: {resolved}")
    if resolved.parent.exists() and not resolved.parent.is_dir():
        raise PreviewRenderError(f"{field} parent is not a directory: {resolved.parent}")
    return resolved


def _validated_source_provider(source_provider: str) -> SourceProvider:
    if source_provider == "kimodo":
        return "kimodo"
    if source_provider == "gem-x":
        return "gem-x"
    raise PreviewRenderError("source_provider must be 'kimodo' or 'gem-x'")


def _validate_request(
    *,
    robot_id: str,
    qpos: ArrayLike,
    fps: float,
    motion_name: str | None,
    contacts: PreviewContactState | None,
    video_path: Path | None,
    thumbnail_path: Path | None,
    width: int,
    height: int,
) -> tuple[
    RobotSpec,
    FloatArray,
    float,
    str | None,
    PreviewContactState | None,
    Path | None,
    Path | None,
]:
    spec = get_robot(robot_id)
    trajectory = _validated_qpos(qpos, spec=spec)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise PreviewRenderError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise PreviewRenderError("height must be a positive integer")
    if isinstance(fps, bool):
        raise PreviewRenderError("fps must be finite and positive")
    try:
        output_fps = float(fps)
    except (TypeError, ValueError) as error:
        raise PreviewRenderError("fps must be finite and positive") from error
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise PreviewRenderError("fps must be finite and positive")
    if video_path is None and thumbnail_path is None:
        raise PreviewRenderError("video_path or thumbnail_path must be provided")
    if video_path is not None and (width % 2 or height % 2):
        raise PreviewRenderError("video width and height must be even for H.264")

    video = (
        None
        if video_path is None
        else _validated_output_path(video_path, suffix=".mp4", field="video_path")
    )
    thumbnail = (
        None
        if thumbnail_path is None
        else _validated_output_path(
            thumbnail_path,
            suffix=".png",
            field="thumbnail_path",
        )
    )
    if video is not None and thumbnail is not None and video == thumbnail:
        raise PreviewRenderError("video_path and thumbnail_path must be different")
    if (motion_name is None) != (contacts is None):
        raise PreviewRenderError(
            "motion_name and contacts must be supplied together for the contact overlay"
        )
    motion_label: str | None = None
    if motion_name is not None and contacts is not None:
        motion_label = str(motion_name).strip()
        if not motion_label:
            raise PreviewRenderError("motion_name must not be empty")
        if contacts.frame_count != len(trajectory):
            raise PreviewRenderError(
                "contact overlay frame count does not match the qpos trajectory"
            )
        if not math.isclose(contacts.fps, output_fps, rel_tol=0.0, abs_tol=1e-9):
            raise PreviewRenderError("contact overlay FPS does not match preview FPS")
    return spec, trajectory, output_fps, motion_label, contacts, video, thumbnail


def _load_imageio(*, require_ffmpeg: bool) -> ModuleType:
    try:
        imageio = importlib.import_module("imageio.v2")
        if require_ffmpeg:
            importlib.import_module("imageio_ffmpeg")
    except ModuleNotFoundError as error:
        requirement = "imageio and imageio-ffmpeg" if require_ffmpeg else "imageio"
        raise PreviewRenderError(
            f"Preview rendering requires {requirement}; install CoRe's video extra"
        ) from error
    return imageio


def _load_scene(spec: RobotSpec) -> tuple[ModuleType, Any, Any]:
    try:
        mujoco = importlib.import_module("mujoco")
    except ModuleNotFoundError as error:
        raise PreviewRenderError("Preview rendering requires MuJoCo") from error

    scene_path = (root_path() / spec.scene_relpath).resolve()
    if not scene_path.is_file():
        raise PreviewRenderError(f"Bundled robot scene is missing: {scene_path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
    except Exception as error:
        raise PreviewRenderError(
            f"Could not load the bundled {spec.robot_id} MuJoCo scene"
        ) from error
    actual = (int(model.nq), int(model.nv), int(model.nu))
    expected = (spec.expected_nq, spec.expected_nv, spec.expected_nu)
    if actual != expected:
        raise PreviewRenderError(
            f"Bundled {spec.robot_id} scene dimensions {actual} do not match {expected}"
        )
    return mujoco, model, data


def preview_camera_for_robot(robot_id: str) -> PreviewCamera:
    """Return the fixed root-following camera used by the established videos."""

    spec = get_robot(robot_id)
    distance, lookat_z = _CAMERA_PRESETS[spec.robot_id]
    return PreviewCamera(
        tracking="root_xy_fixed_z",
        lookat_z=lookat_z,
        distance=float(distance),
        azimuth=_DEFAULT_AZIMUTH,
        elevation=_DEFAULT_ELEVATION,
    )


def _wrap_degrees(angle: float) -> float:
    """Wrap an angle to MuJoCo's conventional ``[-180, 180)`` range."""

    return float((angle + 180.0) % 360.0 - 180.0)


def _gemx_camera_azimuth(
    *,
    mujoco: ModuleType,
    model: Any,
    data: Any,
    spec: RobotSpec,
    initial_qpos: FloatArray,
) -> float:
    """Infer GEM-X's front-quarter camera from the first retargeted pose.

    GEM-X and KiMoDo use different source-facing conventions.  The GEM-X
    notebooks recover the robot's forward direction from the ankle-to-toe
    vectors, point the camera back towards the robot, and add a 25-degree
    three-quarter offset.  Any unavailable or degenerate kinematic data falls
    back to the established KiMoDo camera instead of making preview rendering
    fail.
    """

    try:
        forward_pose(mujoco, model, data, initial_qpos)
        body_names = get_dmr_profile(spec.robot_id).joi_bodies
        body_ids: dict[str, int] = {}
        for key in ("la", "lt", "ra", "rt"):
            body_id = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_names[key],
                )
            )
            if body_id < 0 or body_id >= int(model.nbody):
                return _DEFAULT_AZIMUTH
            body_ids[key] = body_id

        left = np.asarray(
            data.xpos[body_ids["lt"], :2] - data.xpos[body_ids["la"], :2],
            dtype=np.float64,
        )
        right = np.asarray(
            data.xpos[body_ids["rt"], :2] - data.xpos[body_ids["ra"], :2],
            dtype=np.float64,
        )
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if (
            not math.isfinite(left_norm)
            or not math.isfinite(right_norm)
            or left_norm <= 1.0e-9
            or right_norm <= 1.0e-9
        ):
            return _DEFAULT_AZIMUTH

        facing = left / left_norm + right / right_norm
        facing_norm = float(np.linalg.norm(facing))
        if not math.isfinite(facing_norm) or facing_norm <= 1.0e-9:
            return _DEFAULT_AZIMUTH
        heading = math.degrees(math.atan2(float(facing[1]), float(facing[0])))
        return _wrap_degrees(heading + 180.0 + 25.0)
    except Exception:
        return _DEFAULT_AZIMUTH


def _temporary_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=output.suffix,
        delete=False,
    ) as stream:
        return Path(stream.name)


def _publish_temporaries(files: Mapping[Path, Path]) -> None:
    """Atomically publish a group that previously had no destination files."""

    published: list[Path] = []
    try:
        for temporary, output in files.items():
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise PreviewRenderError(f"Renderer produced an empty preview: {output.name}")
            if output.exists():
                raise PreviewRenderError(f"Preview output appeared during rendering: {output}")
            temporary.chmod(0o644)
            os.replace(temporary, output)
            published.append(output)
    except BaseException:
        for output in published:
            output.unlink(missing_ok=True)
        raise


def _render_failure_message(*, robot_id: str, error: Exception) -> str:
    """Return a concise backend failure with an actionable macOS hint when known."""

    detail = f"{type(error).__name__}: {error}"
    message = f"Could not render the {robot_id} preview: {detail}"
    if "invalid coregraphics connection" in str(error).casefold():
        message += (
            ". On macOS, run CoRe from a normal logged-in terminal session with "
            "WindowServer access; CGL rendering is unavailable in a truly headless, "
            "SSH, or daemon session"
        )
    return message


def _render_preview_files(
    *,
    spec: RobotSpec,
    qpos: FloatArray,
    fps: float,
    motion_name: str | None,
    contacts: PreviewContactState | None,
    video_path: Path | None,
    thumbnail_path: Path | None,
    width: int,
    height: int,
    progress: RenderProgress | None = None,
    source_provider: SourceProvider = "kimodo",
) -> PreviewCamera:
    temporary_by_output: dict[Path, Path] = {}
    renderer = None
    writer = None
    try:
        imageio = _load_imageio(require_ffmpeg=video_path is not None)
        mujoco, model, data = _load_scene(spec)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        camera_fit = preview_camera_for_robot(spec.robot_id)
        if source_provider == "gem-x":
            gemx_azimuth = _gemx_camera_azimuth(
                mujoco=mujoco,
                model=model,
                data=data,
                spec=spec,
                initial_qpos=qpos[0],
            )
            camera_fit = PreviewCamera(
                tracking=camera_fit.tracking,
                lookat_z=camera_fit.lookat_z,
                distance=camera_fit.distance,
                azimuth=gemx_azimuth,
                elevation=camera_fit.elevation,
            )
        overlay_runtime = (
            load_overlay_runtime(width=width, height=height) if contacts is not None else None
        )
        contact_scene = (
            prepare_contact_scene(
                mujoco=mujoco,
                model=model,
                data=data,
                robot_id=spec.robot_id,
                qpos=qpos,
                contacts=contacts,
            )
            if contacts is not None
            else None
        )

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = camera_fit.azimuth
        camera.elevation = camera_fit.elevation
        camera.distance = camera_fit.distance
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[1] = 1
        scene_option.geomgroup[2] = 1

        outputs = tuple(path for path in (video_path, thumbnail_path) if path is not None)
        for output in outputs:
            temporary_by_output[output] = _temporary_path(output)
        renderer = mujoco.Renderer(model, height=height, width=width)
        active_renderer = renderer

        def render_frame(index: int) -> NDArray[np.uint8]:
            pose = qpos[index]
            forward_pose(mujoco, model, data, pose)
            camera.lookat[:] = (pose[0], pose[1], camera_fit.lookat_z)
            active_renderer.update_scene(data, camera=camera, scene_option=scene_option)
            if contacts is not None and contact_scene is not None:
                append_contact_markers(
                    mujoco=mujoco,
                    scene=active_renderer.scene,
                    data=data,
                    tick=index,
                    contacts=contacts,
                    state=contact_scene,
                )
            rgb = np.asarray(active_renderer.render())
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise PreviewRenderError(
                    f"MuJoCo returned unexpected RGB frame {rgb.shape}, {rgb.dtype}"
                )
            if contacts is not None and overlay_runtime is not None and motion_name is not None:
                rgb = annotate_frame(
                    rgb,
                    motion_name=motion_name,
                    tick=index,
                    contacts=contacts,
                    runtime=overlay_runtime,
                )
            return rgb

        if video_path is not None:
            writer = imageio.get_writer(
                temporary_by_output[video_path],
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                pixelformat="yuv420p",
                output_params=["-crf", "18", "-preset", "medium"],
            )
            for index in range(len(qpos)):
                writer.append_data(render_frame(index))
                if progress is not None:
                    progress(index + 1, len(qpos))
            writer.close()
            writer = None

        if thumbnail_path is not None:
            thumbnail_frame = len(qpos) // 2
            imageio.imwrite(
                temporary_by_output[thumbnail_path],
                render_frame(thumbnail_frame),
                format="PNG",
            )

        if writer is not None:
            writer.close()
            writer = None
        if renderer is not None:
            renderer.close()
            renderer = None
        _publish_temporaries(
            {temporary: output for output, temporary in temporary_by_output.items()}
        )
    except PreviewRenderError:
        raise
    except Exception as error:
        raise PreviewRenderError(
            _render_failure_message(robot_id=spec.robot_id, error=error)
        ) from error
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        for temporary in temporary_by_output.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return camera_fit


def render_motion_preview(
    *,
    robot_id: str,
    qpos: ArrayLike,
    fps: float,
    motion_name: str | None = None,
    contacts: PreviewContactState | None = None,
    video_path: Path | None,
    thumbnail_path: Path | None,
    width: int = LEGACY_WIDTH,
    height: int = LEGACY_HEIGHT,
    source_provider: SourceProvider = "kimodo",
) -> PreviewArtifacts:
    """Render optional MP4 and PNG previews for one robot trajectory.

    Review callers can provide ``motion_name`` and source-derived ``contacts``
    together to reproduce the established 1280x720 contact visualization. The
    camera follows root X/Y with the robot-specific fixed look-at height used by
    the research videos. KiMoDo keeps the established fixed azimuth, while
    GEM-X derives its azimuth from the first pose's ankle-to-toe direction.
    Existing outputs are never overwritten.
    """

    (
        spec,
        trajectory,
        output_fps,
        motion_label,
        contact_state,
        video,
        thumbnail,
    ) = _validate_request(
        robot_id=robot_id,
        qpos=qpos,
        fps=fps,
        motion_name=motion_name,
        contacts=contacts,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        width=width,
        height=height,
    )
    provider = _validated_source_provider(source_provider)
    camera = _render_preview_files(
        spec=spec,
        qpos=trajectory,
        fps=output_fps,
        motion_name=motion_label,
        contacts=contact_state,
        video_path=video,
        thumbnail_path=thumbnail,
        width=width,
        height=height,
        source_provider=provider,
    )
    return PreviewArtifacts(
        video_path=video,
        thumbnail_path=thumbnail,
        width=width,
        height=height,
        fps=output_fps,
        frame_count=len(trajectory),
        thumbnail_frame=len(trajectory) // 2 if thumbnail is not None else None,
        camera=camera,
        visualization_style=(LEGACY_VISUALIZATION_STYLE if contact_state is not None else None),
        contact_overlay=contact_state is not None,
    )


__all__ = [
    "PreviewArtifacts",
    "PreviewCamera",
    "PreviewRenderError",
    "RenderProgress",
    "SourceProvider",
    "preview_camera_for_robot",
    "render_motion_preview",
]
