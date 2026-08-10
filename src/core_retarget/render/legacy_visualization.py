"""The established CoRe contact visualization used by the research videos.

Pillow is loaded lazily so installing the base package without the ``video``
extra still supports numerical retargeting and NPZ export.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core_retarget.render.contact import (
    CONTACT_COLORS_RGB,
    CONTACT_SHORT_NAMES,
    PreviewContactState,
)
from core_retarget.robots.profiles import get_dmr_profile

LEGACY_VISUALIZATION_STYLE = "legacy-contact-overlay-v1"
LEGACY_WIDTH = 1280
LEGACY_HEIGHT = 720


@dataclass(frozen=True, slots=True)
class _FootContactEvent:
    start: int
    end: int
    anchor: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ContactSceneState:
    """Precomputed world anchors and body IDs used while rendering."""

    foot_events: tuple[tuple[_FootContactEvent, ...], tuple[_FootContactEvent, ...]]
    foot_event_index: NDArray[np.int32]
    contact_age: NDArray[np.int32]
    left_hand_body_id: int
    right_hand_body_id: int


@dataclass(frozen=True, slots=True)
class OverlayRuntime:
    """Lazily imported Pillow modules, fonts, and coordinate scale."""

    image: Any
    image_draw: Any
    font: Any
    font_small: Any
    scale_x: float
    scale_y: float
    font_scale: float


def _contact_runs(label: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    values = np.asarray(label, dtype=bool).reshape(-1)
    padded = np.concatenate([[False], values, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def _load_pillow() -> tuple[ModuleType, ModuleType, ModuleType]:
    try:
        image = importlib.import_module("PIL.Image")
        image_draw = importlib.import_module("PIL.ImageDraw")
        image_font = importlib.import_module("PIL.ImageFont")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Legacy preview visualization requires Pillow; install CoRe's video extra"
        ) from error
    return image, image_draw, image_font


def _font(image_font: ModuleType, size: int) -> Any:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    for path in candidates:
        try:
            return image_font.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return image_font.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return image_font.load_default()


def load_overlay_runtime(*, width: int, height: int) -> OverlayRuntime:
    """Prepare fonts and scaling for a legacy 1280x720 coordinate canvas."""

    image, image_draw, image_font = _load_pillow()
    scale_x = float(width) / LEGACY_WIDTH
    scale_y = float(height) / LEGACY_HEIGHT
    font_scale = min(scale_x, scale_y)
    return OverlayRuntime(
        image=image,
        image_draw=image_draw,
        font=_font(image_font, max(1, int(round(22 * font_scale)))),
        font_small=_font(image_font, max(1, int(round(16 * font_scale)))),
        scale_x=scale_x,
        scale_y=scale_y,
        font_scale=font_scale,
    )


def _xy(runtime: OverlayRuntime, x: int, y: int) -> tuple[int, int]:
    return int(round(x * runtime.scale_x)), int(round(y * runtime.scale_y))


def _box(
    runtime: OverlayRuntime,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int, int]:
    return (*_xy(runtime, x0, y0), *_xy(runtime, x1, y1))


def _line_width(runtime: OverlayRuntime, width: int) -> int:
    return max(1, int(round(width * runtime.font_scale)))


def _draw_contact_timeline(
    draw: Any,
    contacts: PreviewContactState,
    tick: int,
    runtime: OverlayRuntime,
    *,
    x0: int = 66,
    x1: int = 456,
    first_y: int = 222,
) -> None:
    frame_count = contacts.frame_count
    lane_height = 10
    lane_step = 18

    def frame_x(frame_edge: int) -> int:
        return x0 + int(round(frame_edge / max(frame_count, 1) * (x1 - x0)))

    for channel, short_name in enumerate(CONTACT_SHORT_NAMES):
        y0 = first_y + channel * lane_step
        y1 = y0 + lane_height
        color = CONTACT_COLORS_RGB[channel]
        draw.text(
            _xy(runtime, 32, y0 - 6),
            short_name,
            font=runtime.font_small,
            fill=(*color, 255),
        )
        draw.rectangle(
            _box(runtime, x0, y0, x1, y1),
            fill=(35, 39, 47, 210),
            outline=(120, 125, 135, 150),
            width=_line_width(runtime, 1),
        )
        if not contacts.availability[channel]:
            for hatch_x in range(x0 - lane_height, x1, 14):
                draw.line(
                    (*_xy(runtime, hatch_x, y1), *_xy(runtime, hatch_x + lane_height, y0)),
                    fill=(145, 150, 160, 100),
                    width=_line_width(runtime, 1),
                )
            continue
        for start, end in _contact_runs(contacts.labels[:, channel]):
            run_x0, run_x1 = frame_x(start), frame_x(end)
            draw.rectangle(
                _box(runtime, run_x0, y0, max(run_x0 + 1, run_x1), y1),
                fill=(*color, 205),
            )
            center_y = (y0 + y1) // 2
            if channel < 2:
                draw.ellipse(
                    _box(runtime, run_x0 - 4, center_y - 4, run_x0 + 4, center_y + 4),
                    fill=(*color, 255),
                    outline=(255, 255, 255, 230),
                    width=_line_width(runtime, 1),
                )
            else:
                draw.polygon(
                    tuple(
                        _xy(runtime, x, y)
                        for x, y in (
                            (run_x0, center_y - 5),
                            (run_x0 + 5, center_y),
                            (run_x0, center_y + 5),
                            (run_x0 - 5, center_y),
                        )
                    ),
                    fill=(*color, 255),
                )

    playhead_x = frame_x(min(frame_count, tick + 1))
    last_y = first_y + 3 * lane_step + lane_height
    draw.line(
        (*_xy(runtime, playhead_x, first_y - 4), *_xy(runtime, playhead_x, last_y + 4)),
        fill=(255, 255, 255, 245),
        width=_line_width(runtime, 2),
    )


def annotate_frame(
    rgb: NDArray[np.uint8],
    *,
    motion_name: str,
    tick: int,
    contacts: PreviewContactState,
    runtime: OverlayRuntime,
) -> NDArray[np.uint8]:
    """Draw the established motion/contact panel without mutating ``rgb``."""

    if rgb.ndim != 3 or rgb.shape[2:] != (3,) or rgb.dtype != np.uint8:
        raise ValueError(f"RGB frame must have shape (height, width, 3), uint8; found {rgb.shape}")
    if not 0 <= tick < contacts.frame_count:
        raise IndexError(f"Preview tick {tick} is outside {contacts.frame_count} frames")
    image = runtime.image.fromarray(rgb)
    draw = runtime.image_draw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        _box(runtime, 18, 16, 486, 304),
        radius=max(1, int(round(10 * runtime.font_scale))),
        fill=(0, 0, 0, 155),
    )
    draw.text(
        _xy(runtime, 32, 27),
        motion_name,
        font=runtime.font,
        fill=(255, 255, 255, 255),
    )
    draw.text(
        _xy(runtime, 32, 58),
        f"frame {tick}/{contacts.frame_count - 1}   time {contacts.seconds[tick]:.2f}s",
        font=runtime.font,
        fill=(225, 225, 225, 255),
    )
    if contacts.flight[tick]:
        draw.rounded_rectangle(
            _box(runtime, 351, 52, 456, 80),
            radius=max(1, int(round(7 * runtime.font_scale))),
            fill=(125, 88, 5, 220),
        )
        draw.text(
            _xy(runtime, 370, 54),
            "FLIGHT",
            font=runtime.font_small,
            fill=(255, 220, 95, 255),
        )

    matching = np.flatnonzero(
        (contacts.segment_ranges[:, 0] <= tick) & (tick < contacts.segment_ranges[:, 1])
    )
    segment_index = int(matching[0]) if len(matching) else len(contacts.segment_ranges) - 1
    phase_start = float(contacts.segment_boundaries[segment_index])
    phase_end = float(contacts.segment_boundaries[segment_index + 1])
    phase_close = "]" if segment_index == len(contacts.segment_ranges) - 1 else ")"
    draw.text(
        _xy(runtime, 32, 89),
        f"contact segment {segment_index + 1}/{len(contacts.segment_ranges)}   "
        f"phase [{phase_start:.3f}, {phase_end:.3f}{phase_close}",
        font=runtime.font_small,
        fill=(180, 225, 255, 255),
    )

    bar_x0, bar_x1 = 32, 456
    bar_y0, bar_y1 = 119, 133
    segment_colors = (
        (70, 150, 235, 145),
        (75, 205, 150, 145),
        (245, 175, 65, 145),
        (185, 115, 225, 145),
    )
    for index, (boundary_start, boundary_end) in enumerate(
        zip(
            contacts.segment_boundaries[:-1],
            contacts.segment_boundaries[1:],
            strict=True,
        )
    ):
        x0 = bar_x0 + int(round(float(boundary_start) * (bar_x1 - bar_x0)))
        x1 = bar_x0 + int(round(float(boundary_end) * (bar_x1 - bar_x0)))
        base_color = segment_colors[index % len(segment_colors)]
        alpha = 235 if index == segment_index else base_color[3]
        draw.rectangle(
            _box(runtime, x0, bar_y0, max(x0 + 1, x1), bar_y1),
            fill=(*base_color[:3], alpha),
        )
        if index > 0:
            draw.line(
                (*_xy(runtime, x0, bar_y0 - 2), *_xy(runtime, x0, bar_y1 + 2)),
                fill=(255, 255, 255, 220),
                width=_line_width(runtime, 2),
            )
    progress = (float(tick) + 0.5) / max(float(contacts.frame_count), 1.0)
    marker_x = bar_x0 + int(round(progress * (bar_x1 - bar_x0)))
    draw.line(
        (*_xy(runtime, marker_x, bar_y0 - 5), *_xy(runtime, marker_x, bar_y1 + 5)),
        fill=(255, 255, 255, 255),
        width=_line_width(runtime, 3),
    )
    draw.rectangle(
        _box(runtime, bar_x0, bar_y0, bar_x1, bar_y1),
        outline=(255, 255, 255, 180),
        width=_line_width(runtime, 1),
    )

    draw.text(
        _xy(runtime, 32, 148),
        "CONTACT STATE",
        font=runtime.font_small,
        fill=(220, 225, 235, 255),
    )
    chip_y0, chip_y1 = 171, 204
    chip_width, chip_gap = 102, 4
    for channel, short_name in enumerate(CONTACT_SHORT_NAMES):
        x0 = 32 + channel * (chip_width + chip_gap)
        x1 = x0 + chip_width
        color = CONTACT_COLORS_RGB[channel]
        available = bool(contacts.availability[channel])
        active = bool(contacts.labels[tick, channel]) if available else False
        if not available:
            fill, outline, state_text = (55, 58, 65, 210), (135, 140, 150, 170), "N/A"
        elif active:
            fill, outline, state_text = (*color, 205), (*color, 255), "ON"
        else:
            fill, outline, state_text = (25, 29, 36, 215), (*color, 155), "OFF"
        draw.rounded_rectangle(
            _box(runtime, x0, chip_y0, x1, chip_y1),
            radius=max(1, int(round(7 * runtime.font_scale))),
            fill=fill,
            outline=outline,
            width=_line_width(runtime, 2),
        )
        if available and channel < 2:
            text = f"{short_name} {state_text}  {contacts.confidence[tick, channel]:.2f}"
        else:
            text = f"{short_name} {state_text}"
        draw.text(
            _xy(runtime, x0 + 5, chip_y0 + 6),
            text,
            font=runtime.font_small,
            fill=(255, 255, 255, 255),
        )

    _draw_contact_timeline(draw, contacts, tick, runtime)
    return np.asarray(image, dtype=np.uint8)


def forward_pose(mujoco: ModuleType, model: Any, data: Any, qpos: NDArray[np.float64]) -> None:
    """Match the legacy renderer's limited-joint clipping before FK."""

    data.qpos[:] = qpos
    for joint_id in range(int(model.njnt)):
        if not bool(model.jnt_limited[joint_id]):
            continue
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        data.qpos[address] = np.clip(data.qpos[address], float(lower), float(upper))
    mujoco.mj_forward(model, data)


def _body_id(mujoco: ModuleType, model: Any, name: str) -> int:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise ValueError(f"Legacy visualization body is missing: {name}")
    return body_id


def floor_height(mujoco: ModuleType, model: Any) -> float:
    """Return the first world-attached plane height from the packaged scene."""

    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) == plane and int(model.geom_bodyid[geom_id]) == 0:
            return float(model.geom_pos[geom_id, 2])
    raise ValueError("Legacy visualization requires a world-attached floor plane")


def prepare_contact_scene(
    *,
    mujoco: ModuleType,
    model: Any,
    data: Any,
    robot_id: str,
    qpos: NDArray[np.float64],
    contacts: PreviewContactState,
) -> ContactSceneState:
    """Precompute contact-run anchors from robot toe-body trajectories."""

    profile = get_dmr_profile(robot_id)
    left_toe_id = _body_id(mujoco, model, profile.joi_bodies["lt"])
    right_toe_id = _body_id(mujoco, model, profile.joi_bodies["rt"])
    left_hand_name = profile.joi_bodies.get("lh", profile.joi_bodies["lw"])
    right_hand_name = profile.joi_bodies.get("rh", profile.joi_bodies["rw"])
    left_hand_id = _body_id(mujoco, model, left_hand_name)
    right_hand_id = _body_id(mujoco, model, right_hand_name)

    toe_positions = np.zeros((len(qpos), 2, 3), dtype=np.float64)
    for tick, pose in enumerate(qpos):
        forward_pose(mujoco, model, data, pose)
        toe_positions[tick, 0] = data.xpos[left_toe_id]
        toe_positions[tick, 1] = data.xpos[right_toe_id]

    z_floor = floor_height(mujoco, model)
    event_groups: list[tuple[_FootContactEvent, ...]] = []
    event_index = np.full((len(qpos), 2), -1, dtype=np.int32)
    for channel in range(2):
        events: list[_FootContactEvent] = []
        for start, end in _contact_runs(contacts.labels[:, channel]):
            anchor = np.median(toe_positions[start:end, channel], axis=0)
            anchor[2] = z_floor
            event_index[start:end, channel] = len(events)
            anchor.setflags(write=False)
            events.append(_FootContactEvent(start=start, end=end, anchor=anchor))
        event_groups.append(tuple(events))

    contact_age = np.zeros_like(contacts.labels, dtype=np.int32)
    for channel in range(4):
        for start, end in _contact_runs(contacts.labels[:, channel]):
            contact_age[start:end, channel] = np.arange(end - start, dtype=np.int32)
    event_index.setflags(write=False)
    contact_age.setflags(write=False)
    return ContactSceneState(
        foot_events=(event_groups[0], event_groups[1]),
        foot_event_index=event_index,
        contact_age=contact_age,
        left_hand_body_id=left_hand_id,
        right_hand_body_id=right_hand_id,
    )


def _append_scene_geom(
    mujoco: ModuleType,
    scene: Any,
    geom_type: Any,
    size: tuple[float, float, float],
    position: NDArray[np.float64],
    rgba: tuple[float, float, float, float],
    rotation: NDArray[np.float64] | None = None,
) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    matrix = np.eye(3, dtype=np.float64) if rotation is None else rotation
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.asarray(matrix, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def _append_foot_stamp(
    mujoco: ModuleType,
    scene: Any,
    *,
    position: NDArray[np.float64],
    color_rgb: tuple[int, int, int],
    confidence: float,
    contact_age: int,
) -> None:
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    pulse = max(0.0, 1.0 - float(contact_age) / 5.0)
    radius = 0.075 + 0.014 * pulse
    alpha = 0.30 + 0.65 * float(np.clip(confidence, 0.0, 1.0))
    center = np.asarray(position, dtype=np.float64).copy()
    center[2] += 0.0025
    _append_scene_geom(
        mujoco,
        scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (radius, 0.0025, 0.0),
        center,
        (*color, min(0.95, alpha)),
    )
    center[2] += 0.0008
    _append_scene_geom(
        mujoco,
        scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.72 * radius, 0.0025, 0.0),
        center,
        (*color, 0.18 + 0.30 * float(np.clip(confidence, 0.0, 1.0))),
    )


def _append_hand_halo(
    mujoco: ModuleType,
    scene: Any,
    *,
    position: NDArray[np.float64],
    color_rgb: tuple[int, int, int],
    contact_age: int,
) -> None:
    color = np.asarray(color_rgb, dtype=np.float64) / 255.0
    pulse = max(0.0, 1.0 - float(contact_age) / 5.0)
    radius = 0.058 + 0.010 * pulse
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    center = np.asarray(position, dtype=np.float64)
    for angle in angles:
        cosine, sine = np.cos(angle), np.sin(angle)
        for offset in (
            np.asarray([radius * cosine, radius * sine, 0.0]),
            np.asarray([radius * cosine, 0.0, radius * sine]),
        ):
            _append_scene_geom(
                mujoco,
                scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (0.0075, 0.0075, 0.0075),
                center + offset,
                (*color, 0.85),
            )
    _append_scene_geom(
        mujoco,
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.016, 0.016, 0.016),
        center,
        (*color, 0.95),
    )


def append_contact_markers(
    *,
    mujoco: ModuleType,
    scene: Any,
    data: Any,
    tick: int,
    contacts: PreviewContactState,
    state: ContactSceneState,
) -> None:
    """Append floor stamps and optional hand halos to an updated MuJoCo scene."""

    for channel in range(2):
        event_index = int(state.foot_event_index[tick, channel])
        if event_index >= 0:
            event = state.foot_events[channel][event_index]
            _append_foot_stamp(
                mujoco,
                scene,
                position=event.anchor,
                color_rgb=CONTACT_COLORS_RGB[channel],
                confidence=float(contacts.confidence[tick, channel]),
                contact_age=int(state.contact_age[tick, channel]),
            )
    for channel, body_id in (
        (2, state.left_hand_body_id),
        (3, state.right_hand_body_id),
    ):
        if contacts.availability[channel] and contacts.labels[tick, channel]:
            _append_hand_halo(
                mujoco,
                scene,
                position=np.asarray(data.xpos[body_id], dtype=np.float64),
                color_rgb=CONTACT_COLORS_RGB[channel],
                contact_age=int(state.contact_age[tick, channel]),
            )


__all__ = [
    "ContactSceneState",
    "LEGACY_HEIGHT",
    "LEGACY_VISUALIZATION_STYLE",
    "LEGACY_WIDTH",
    "OverlayRuntime",
    "annotate_frame",
    "append_contact_markers",
    "floor_height",
    "forward_pose",
    "load_overlay_runtime",
    "prepare_contact_scene",
]
