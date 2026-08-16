"""Safe GEM-X `.pt` loading and conversion to CoRe's SOMA77 world motion.

GEM-X stores SOMA body parameters rather than the already-evaluated joint
arrays used by Kimodo `.npz` files. This module performs fixed-rig forward
kinematics, converts the result to Z-up, and applies time-varying support-floor
normalization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, median_filter  # type: ignore[import-untyped]

from rimkit.assets import root_path
from rimkit.exceptions import MotionValidationError
from rimkit.motion.soma import (
    DEFAULT_FPS,
    SOMA_JOINT_COUNT,
    SomaMotion,
    SomaMotionSummary,
    rotation_x,
)
from rimkit.motion.soma_joints import SOMA77_JOINT_INDEX, SOMA77_JOINT_NAMES

GEMX_STATIC_CONTACT_NAMES = (
    "left_ankle",
    "left_foot",
    "right_ankle",
    "right_foot",
    "left_wrist",
    "right_wrist",
)


def _readonly(array: NDArray[Any], *, dtype: Any) -> NDArray[Any]:
    value = np.array(array, dtype=dtype, copy=True, order="C")
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class GemxContactSeed:
    """GEM-X toe-contact labels retained across source normalization."""

    left_raw: NDArray[np.bool_]
    right_raw: NDArray[np.bool_]
    left_fused: NDArray[np.bool_]
    right_fused: NDArray[np.bool_]
    left_confidence: NDArray[np.float64]
    right_confidence: NDArray[np.float64]
    floor_z: NDArray[np.float64]

    def __post_init__(self) -> None:
        frame_count = len(self.left_raw)
        for name in ("left_raw", "right_raw", "left_fused", "right_fused"):
            value = _readonly(getattr(self, name), dtype=np.bool_)
            if value.shape != (frame_count,):
                raise MotionValidationError(f"GEM-X {name} must have shape ({frame_count},).")
            object.__setattr__(self, name, value)
        for name in ("left_confidence", "right_confidence", "floor_z"):
            value = _readonly(getattr(self, name), dtype=np.float64)
            if value.shape != (frame_count,) or not np.isfinite(value).all():
                raise MotionValidationError(
                    f"GEM-X {name} must be a finite array with shape ({frame_count},)."
                )
            object.__setattr__(self, name, value)


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_fps(fps_override: float | None) -> float:
    try:
        fps = DEFAULT_FPS if fps_override is None else float(fps_override)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionValidationError("FPS must be a finite real scalar.") from exc
    if not np.isfinite(fps) or not 0.0 < fps <= 1000.0:
        raise MotionValidationError(f"FPS must be in (0, 1000], found {fps}.")
    return fps


def _as_numpy(value: Any, *, name: str) -> NDArray[Any]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
    except Exception as exc:
        raise MotionValidationError(f"Could not convert GEM-X {name} to an array.") from exc
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise MotionValidationError(f"GEM-X {name} must have a real numeric dtype.")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.isfinite(array).all():
        raise MotionValidationError(f"GEM-X {name} contains non-real, NaN, or infinite values.")
    return array


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        raise MotionValidationError(
            "GEM-X .pt input requires PyTorch; install RIMKit with the 'gemx' extra."
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise MotionValidationError(
            "This PyTorch version cannot perform weights-only .pt loading; upgrade PyTorch."
        ) from exc
    except Exception as exc:
        raise MotionValidationError(f"Could not safely open GEM-X .pt: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MotionValidationError("GEM-X .pt root must be a dictionary-like mapping.")
    return payload


def _body_params(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = payload.get("body_params_global")
    if candidate is None:
        net_outputs = payload.get("net_outputs")
        if isinstance(net_outputs, Mapping):
            candidate = net_outputs.get("pred_body_params_global")
    if not isinstance(candidate, Mapping):
        raise MotionValidationError(
            "GEM-X .pt is missing body_params_global (or net_outputs.pred_body_params_global)."
        )
    return candidate


def _static_logits(payload: Mapping[str, Any], *, frame_count: int) -> NDArray[np.float64]:
    logits: Any = None
    net_outputs = payload.get("net_outputs")
    if isinstance(net_outputs, Mapping):
        logits = net_outputs.get("static_conf_logits")
        if logits is None:
            model_output = net_outputs.get("model_output")
            if isinstance(model_output, Mapping):
                logits = model_output.get("static_conf_logits")
    if logits is None:
        raise MotionValidationError(
            "GEM-X .pt is missing net_outputs.static_conf_logits required for floor/contact fusion."
        )
    values = _as_numpy(logits, name="static_conf_logits").astype(np.float64, copy=False)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (frame_count, len(GEMX_STATIC_CONTACT_NAMES)):
        raise MotionValidationError(
            "GEM-X static_conf_logits must have shape "
            f"({frame_count}, 6) or (1, {frame_count}, 6); found {values.shape}."
        )
    return values


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _rodrigues_batch(rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(values, axis=1)
    matrices = np.tile(np.eye(3, dtype=np.float64), (len(values), 1, 1))
    valid = theta > 1e-12
    if not np.any(valid):
        return matrices
    axis = np.zeros_like(values)
    axis[valid] = values[valid] / theta[valid, None]
    x, y, z = axis[valid].T
    skew = np.zeros((int(np.sum(valid)), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    angles = theta[valid]
    matrices[valid] = (
        np.eye(3, dtype=np.float64)[None]
        + np.sin(angles)[:, None, None] * skew
        + (1.0 - np.cos(angles))[:, None, None] * (skew @ skew)
    )
    return matrices


def _invert_transforms(transforms: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(transforms, dtype=np.float64)
    output = np.tile(np.eye(4, dtype=np.float64), values.shape[:-2] + (1, 1))
    rotations = values[..., :3, :3]
    positions = values[..., :3, 3]
    rotation_t = np.swapaxes(rotations, -1, -2)
    output[..., :3, :3] = rotation_t
    output[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation_t, positions)
    return output


def _skin_rig() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    path = root_path() / "soma" / "soma77_bind_rig.npz"
    if not path.is_file():
        raise MotionValidationError(f"Packaged SOMA77 rig asset is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            bind = np.asarray(archive["bind_rig_transform"], dtype=np.float64)
            names = tuple(np.asarray(archive["rig_joint_names"]).astype(str).tolist())
            connections = np.asarray(archive["rig_joint_connections"], dtype=np.int64)
    except Exception as exc:
        raise MotionValidationError(f"Could not load packaged SOMA77 rig: {exc}") from exc
    if bind.shape != (SOMA_JOINT_COUNT, 4, 4):
        raise MotionValidationError(f"SOMA77 bind transforms have invalid shape {bind.shape}.")
    if names != SOMA77_JOINT_NAMES:
        raise MotionValidationError("Packaged SOMA77 rig joint order does not match RIMKit.")
    if connections.ndim != 2 or connections.shape[1] != 2:
        raise MotionValidationError("Packaged SOMA77 rig connections are invalid.")
    return bind, connections


def _forward_kinematics(
    body_params: Mapping[str, Any],
    *,
    max_frames: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    missing = sorted({"body_pose", "global_orient", "transl"} - set(body_params))
    if missing:
        raise MotionValidationError(f"GEM-X body parameters are missing: {', '.join(missing)}.")
    body_pose = _as_numpy(body_params["body_pose"], name="body_pose")
    global_orient = _as_numpy(body_params["global_orient"], name="global_orient")
    translation = _as_numpy(body_params["transl"], name="transl")
    # GEM-X's top-level body parameters are commonly unbatched, while
    # ``net_outputs.pred_body_params_global`` retains its singleton inference
    # batch. Normalize both representations before enforcing the timeline
    # shapes below.
    if body_pose.ndim == 3 and body_pose.shape[0] == 1 and body_pose.shape[2] == 76 * 3:
        body_pose = body_pose[0]
    elif body_pose.ndim == 4 and body_pose.shape[0] == 1 and body_pose.shape[2:] == (76, 3):
        body_pose = body_pose[0]
    if global_orient.ndim in {3, 4} and global_orient.shape[0] == 1:
        global_orient = global_orient[0]
    if translation.ndim == 3 and translation.shape[0] == 1:
        translation = translation[0]
    if body_pose.ndim == 2 and body_pose.shape[1] == 76 * 3:
        frame_count = int(body_pose.shape[0])
        body_pose = body_pose.reshape(frame_count, 76, 3)
    elif body_pose.ndim == 3 and body_pose.shape[1:] == (76, 3):
        frame_count = int(body_pose.shape[0])
    else:
        raise MotionValidationError(
            "GEM-X body_pose must have shape (T, 228), (T, 76, 3), or the same "
            f"with a singleton batch axis; found {body_pose.shape}."
        )
    if frame_count < 2:
        raise MotionValidationError("GEM-X retargeting requires at least two frames.")
    if frame_count > max_frames:
        raise MotionValidationError(
            f"Motion has {frame_count} frames, exceeding the limit of {max_frames}."
        )
    if global_orient.shape == (frame_count, 3):
        global_orient = global_orient[:, None, :]
    if global_orient.shape != (frame_count, 1, 3):
        raise MotionValidationError(
            "GEM-X global_orient must have shape (T, 3), (T, 1, 3), or the same "
            f"with a singleton batch axis; found {global_orient.shape}."
        )
    if translation.shape != (frame_count, 3):
        raise MotionValidationError(
            f"GEM-X transl must have shape (T, 3) or (1, T, 3); found {translation.shape}."
        )

    poses = np.concatenate((global_orient, body_pose), axis=1).astype(np.float64, copy=False)
    local_rotations = _rodrigues_batch(poses.reshape(-1, 3)).reshape(
        frame_count, SOMA_JOINT_COUNT, 3, 3
    )
    bind, connections = _skin_rig()
    parents = np.arange(SOMA_JOINT_COUNT, dtype=np.int64)
    for parent, child in connections.reshape(-1, 2):
        parents[int(child)] = int(parent)
    parents[0] = 0
    inverse_bind = _invert_transforms(bind)
    local_bind = bind.copy()
    for joint in range(1, SOMA_JOINT_COUNT):
        local_bind[joint] = inverse_bind[parents[joint]] @ bind[joint]

    local = np.tile(np.eye(4, dtype=np.float64), (frame_count, SOMA_JOINT_COUNT, 1, 1))
    # Body parameters directly define local joint rotations; bind-pose joint
    # orientation is not multiplied into them.
    local[:, :, :3, :3] = local_rotations
    local[:, :, :3, 3] = local_bind[None, :, :3, 3]
    local[:, 0, :3, 3] = translation
    world = np.empty_like(local)
    world[:, 0] = local[:, 0]
    for joint in range(1, SOMA_JOINT_COUNT):
        world[:, joint] = world[:, parents[joint]] @ local[:, joint]
    return world[:, :, :3, 3], world[:, :, :3, :3]


def _odd_window(seconds: float, dt: float) -> int:
    window = max(1, int(round(float(seconds) / max(float(dt), 1e-12))))
    return window if window % 2 else window + 1


def _floor_from_contacts(
    left_z: NDArray[np.float64],
    right_z: NDArray[np.float64],
    left_contact: NDArray[np.bool_],
    right_contact: NDArray[np.bool_],
    *,
    dt: float,
) -> NDArray[np.float64]:
    sample_mask = left_contact | right_contact
    ticks = np.flatnonzero(sample_mask)
    if len(ticks) == 0:
        raise MotionValidationError("Cannot estimate a GEM-X floor without contact samples.")
    samples = np.zeros(len(left_z), dtype=np.float64)
    only_left = left_contact & ~right_contact
    only_right = right_contact & ~left_contact
    both = left_contact & right_contact
    samples[only_left] = left_z[only_left]
    samples[only_right] = right_z[only_right]
    samples[both] = np.minimum(left_z[both], right_z[both])
    floor = np.interp(np.arange(len(samples), dtype=np.float64), ticks, samples[ticks])
    floor = median_filter(floor, size=_odd_window(0.10, dt), mode="nearest")
    sigma = 0.15 / max(dt, 1e-12)
    return np.asarray(
        gaussian_filter1d(floor, sigma=sigma, mode="nearest", truncate=3.0),
        dtype=np.float64,
    )


def _bounded_false_segments(label: NDArray[np.bool_]) -> list[NDArray[np.int64]]:
    false_ticks = np.flatnonzero(~label)
    if len(false_ticks) == 0:
        return []
    segments: list[NDArray[np.int64]] = []
    for candidate in np.split(false_ticks, np.flatnonzero(np.diff(false_ticks) != 1) + 1):
        if len(candidate) and candidate[0] > 0 and candidate[-1] < len(label) - 1:
            segments.append(np.asarray(candidate, dtype=np.int64))
    return segments


def _fuse_contacts(
    seconds: NDArray[np.float64],
    positions: NDArray[np.float64],
    logits: NDArray[np.float64],
) -> GemxContactSeed:
    frame_count = len(seconds)
    dt = float(np.median(np.diff(seconds)))
    left_toe = positions[:, SOMA77_JOINT_INDEX["LeftToeBase"]]
    right_toe = positions[:, SOMA77_JOINT_INDEX["RightToeBase"]]
    # Preserve the source numeric path by materializing logits and sigmoid
    # confidence as float32 before promoting them to float64.
    confidence = _sigmoid(np.asarray(logits, dtype=np.float32)).astype(np.float64)
    left_confidence_raw = np.clip(confidence[:, 1], 0.0, 1.0)
    right_confidence_raw = np.clip(confidence[:, 3], 0.0, 1.0)
    left_raw = (logits[:, 1] > 0.0) | (left_confidence_raw >= 0.50)
    right_raw = (logits[:, 3] > 0.0) | (right_confidence_raw >= 0.50)
    floor_initial = _floor_from_contacts(
        left_toe[:, 2], right_toe[:, 2], left_raw, right_raw, dt=dt
    )
    sigma = max(1e-6, 0.5 * 0.15 / dt)
    left_smooth = gaussian_filter1d(left_toe, sigma=sigma, axis=0, mode="nearest")
    right_smooth = gaussian_filter1d(right_toe, sigma=sigma, axis=0, mode="nearest")
    left_speed = np.mean(np.abs(np.gradient(left_smooth, seconds, axis=0)), axis=1)
    right_speed = np.mean(np.abs(np.gradient(right_smooth, seconds, axis=0)), axis=1)
    left_clearance = np.abs(left_toe[:, 2] - floor_initial)
    right_clearance = np.abs(right_toe[:, 2] - floor_initial)

    def fuse(
        raw: NDArray[np.bool_],
        toe: NDArray[np.float64],
        clearance: NDArray[np.float64],
        speed: NDArray[np.float64],
        source_confidence: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        output = raw.copy()
        geometry = (clearance <= 0.035) & (speed <= 0.25)
        maximum_frames = max(1, int(round(0.35 / max(dt, 1e-12))))
        for segment in _bounded_false_segments(output):
            if len(segment) > maximum_frames:
                continue
            if output[segment[0] - 1] and output[segment[-1] + 1]:
                if (
                    float(np.mean(geometry[segment])) >= 0.60
                    and float(np.ptp(toe[segment, 2])) <= 0.04
                ):
                    output[segment] = True
        active = False
        for tick in range(frame_count):
            if raw[tick]:
                active = True
            elif active and source_confidence[tick] >= 0.20 and geometry[tick]:
                output[tick] = True
            else:
                active = False
        geometry_confidence = np.clip(1.0 - clearance / 0.035, 0.0, 1.0)
        fused_confidence = np.where(
            output,
            np.maximum(source_confidence, geometry_confidence),
            source_confidence,
        )
        return output, fused_confidence

    left_fused, left_confidence = fuse(
        left_raw, left_toe, left_clearance, left_speed, left_confidence_raw
    )
    right_fused, right_confidence = fuse(
        right_raw, right_toe, right_clearance, right_speed, right_confidence_raw
    )
    floor = _floor_from_contacts(left_toe[:, 2], right_toe[:, 2], left_fused, right_fused, dt=dt)
    return GemxContactSeed(
        left_raw=left_raw,
        right_raw=right_raw,
        left_fused=left_fused,
        right_fused=right_fused,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        floor_z=floor,
    )


def load_gemx_motion(
    path: str | Path,
    *,
    fps_override: float | None = None,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
    max_frames: int = 1_000_000,
) -> tuple[SomaMotion, GemxContactSeed]:
    """Load a GEM-X `.pt` using weights-only PyTorch deserialization."""

    motion_path = Path(path).expanduser().resolve()
    if not motion_path.is_file():
        raise MotionValidationError(f"Motion file does not exist: {motion_path}")
    if motion_path.suffix.lower() != ".pt":
        raise MotionValidationError(f"GEM-X input must use the .pt extension: {motion_path.name}")
    if motion_path.stat().st_size > max_file_bytes:
        raise MotionValidationError(
            f"Motion file is larger than the {max_file_bytes}-byte safety limit."
        )
    fps = _resolve_fps(fps_override)
    payload = _safe_torch_load(motion_path)
    positions, rotations = _forward_kinematics(_body_params(payload), max_frames=max_frames)
    # The explained notebooks materialize the parser result as float32 before
    # constructing their source chains. Preserve that numerical boundary, then
    # continue in CoRe's float64 stage arithmetic.
    positions = positions.astype(np.float32).astype(np.float64)
    rotations = rotations.astype(np.float32).astype(np.float64)
    frame_count = int(positions.shape[0])
    logits = _static_logits(payload, frame_count=frame_count)

    world_rotation = rotation_x(np.pi / 2.0)
    positions = np.einsum("ij,tkj->tki", world_rotation, positions)
    rotations = np.einsum("ij,tkjl->tkil", world_rotation, rotations)
    seconds = np.arange(frame_count, dtype=np.float64) / fps
    seed = _fuse_contacts(seconds, positions, logits)
    positions = positions.copy()
    positions[:, :, 2] -= seed.floor_z[:, None]

    # Preserve fused GEM-X toe support in the established six-channel in-memory
    # contract.  Channel placement intentionally follows the Kimodo adapter's
    # left [1:3] / right [4:6] interpretation; raw GEM-X channel order is never
    # exposed as if it were Kimodo.
    contacts = np.zeros((frame_count, 6), dtype=np.bool_)
    contacts[:, 1] = seed.left_fused
    contacts[:, 4] = seed.right_fused
    warnings = (
        ("GEM-X .pt does not carry an FPS field; using the default of 30 Hz.",)
        if fps_override is None
        else ()
    )
    summary = SomaMotionSummary(
        path=motion_path,
        sha256=_file_digest(motion_path),
        frame_count=frame_count,
        fps=fps,
        duration_seconds=frame_count / fps,
        keys=tuple(sorted(str(key) for key in payload)),
        contact_channels=6,
        warnings=warnings,
    )
    motion = SomaMotion(
        summary=summary,
        seconds=seconds,
        posed_joints=positions,
        global_rot_mats=rotations,
        foot_contacts=contacts,
        world_rotation=world_rotation,
        position_scale=1.0,
        position_offset=np.zeros(3, dtype=np.float64),
        z_up=True,
    )
    return motion, seed


__all__ = ["GEMX_STATIC_CONTACT_NAMES", "GemxContactSeed", "load_gemx_motion"]
