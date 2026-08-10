"""Safe loading and world-frame conversion for Kimodo SOMA motion."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core_retarget.exceptions import MotionValidationError

DEFAULT_FPS = 30.0
SOMA_JOINT_COUNT = 77


@dataclass(frozen=True)
class SomaMotionSummary:
    """Validated metadata for a SOMA NPZ."""

    path: Path
    sha256: str
    frame_count: int
    fps: float
    duration_seconds: float
    keys: tuple[str, ...]
    contact_channels: int | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SomaMotion:
    """Loaded SOMA motion in the world frame used by the retargeting stages.

    The arrays are defensive ``float64`` copies and are marked read-only so
    stages cannot alter their input in place.
    """

    summary: SomaMotionSummary
    seconds: NDArray[np.float64]
    posed_joints: NDArray[np.float64]
    global_rot_mats: NDArray[np.float64]
    foot_contacts: NDArray[Any] | None
    world_rotation: NDArray[np.float64]
    position_scale: float
    position_offset: NDArray[np.float64]
    z_up: bool

    def __post_init__(self) -> None:
        seconds = _readonly_copy(self.seconds, dtype=np.float64)
        posed_joints = _readonly_copy(self.posed_joints, dtype=np.float64)
        global_rot_mats = _readonly_copy(self.global_rot_mats, dtype=np.float64)
        world_rotation = _readonly_copy(self.world_rotation, dtype=np.float64)
        position_offset = _readonly_copy(self.position_offset, dtype=np.float64)
        contacts = (
            None if self.foot_contacts is None else _readonly_copy(self.foot_contacts, dtype=None)
        )

        frame_count = self.summary.frame_count
        if seconds.shape != (frame_count,):
            raise MotionValidationError(
                f"seconds must have shape ({frame_count},); found {seconds.shape}."
            )
        if posed_joints.shape != (frame_count, SOMA_JOINT_COUNT, 3):
            raise MotionValidationError(
                "posed_joints does not match the validated SOMA motion shape."
            )
        if global_rot_mats.shape != (frame_count, SOMA_JOINT_COUNT, 3, 3):
            raise MotionValidationError(
                "global_rot_mats does not match the validated SOMA motion shape."
            )
        if contacts is not None and contacts.shape[0] != frame_count:
            raise MotionValidationError("foot_contacts does not match the motion frame count.")
        if world_rotation.shape != (3, 3):
            raise MotionValidationError("world_rotation must have shape (3, 3).")
        if position_offset.shape != (3,):
            raise MotionValidationError("position_offset must have shape (3,).")

        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "posed_joints", posed_joints)
        object.__setattr__(self, "global_rot_mats", global_rot_mats)
        object.__setattr__(self, "foot_contacts", contacts)
        object.__setattr__(self, "world_rotation", world_rotation)
        object.__setattr__(self, "position_offset", position_offset)

    @property
    def path(self) -> Path:
        """Resolved path of the source NPZ."""

        return self.summary.path

    @property
    def sha256(self) -> str:
        """SHA-256 identity of the source NPZ."""

        return self.summary.sha256

    @property
    def fps(self) -> float:
        """Sampling rate used to construct ``seconds``."""

        return self.summary.fps

    @property
    def frame_count(self) -> int:
        """Number of motion frames."""

        return self.summary.frame_count

    @property
    def duration_seconds(self) -> float:
        """Frame-count duration used by the existing public input contract."""

        return self.summary.duration_seconds


def _readonly_copy(array: NDArray[Any], *, dtype: Any) -> NDArray[Any]:
    copied = np.array(array, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _file_digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _is_real_numeric(array: NDArray[Any]) -> bool:
    return np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype, np.complexfloating
    )


def _finite_real_array(array: NDArray[Any], *, name: str) -> None:
    if array.dtype.hasobject or not _is_real_numeric(array):
        raise MotionValidationError(f"{name} must have a real numeric dtype.")
    if not np.isfinite(array).all():
        raise MotionValidationError(f"{name} contains NaN or infinity.")


def _resolve_fps(archive: Any, fps_override: float | None, warnings: list[str]) -> float:
    try:
        if fps_override is not None:
            fps = float(fps_override)
        elif "fps" in archive.files:
            fps_array = np.asarray(archive["fps"])
            if fps_array.size != 1:
                raise MotionValidationError("fps must be a scalar when present.")
            _finite_real_array(fps_array, name="fps")
            fps = float(fps_array.reshape(-1)[0])
        else:
            fps = DEFAULT_FPS
            warnings.append("No fps field found; using the Kimodo example default of 30 Hz.")
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionValidationError("FPS must be a finite real scalar.") from exc

    if not np.isfinite(fps) or not 0.0 < fps <= 1000.0:
        raise MotionValidationError(f"FPS must be in (0, 1000], found {fps}.")
    return fps


def validate_soma_npz(
    path: str | Path,
    *,
    fps_override: float | None = None,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
    max_frames: int = 1_000_000,
) -> SomaMotionSummary:
    """Validate a SOMA NPZ without enabling NumPy pickle loading."""

    motion_path = Path(path).expanduser().resolve()
    if not motion_path.is_file():
        raise MotionValidationError(f"Motion file does not exist: {motion_path}")
    if motion_path.stat().st_size > max_file_bytes:
        raise MotionValidationError(
            f"Motion file is larger than the {max_file_bytes}-byte safety limit."
        )

    warnings: list[str] = []
    if motion_path.suffix.lower() != ".npz":
        warnings.append("The file extension is not .npz.")

    try:
        archive = np.load(motion_path, allow_pickle=False)
    except Exception as exc:
        raise MotionValidationError(f"Could not open SOMA NPZ: {exc}") from exc

    with archive:
        keys = tuple(sorted(archive.files))
        for key in keys:
            try:
                candidate = np.asarray(archive[key])
            except ValueError as exc:
                raise MotionValidationError(f"{key} is pickle-backed or has object dtype.") from exc
            if candidate.dtype.hasobject:
                raise MotionValidationError(f"{key} has unsupported object dtype.")

        missing = sorted({"posed_joints", "global_rot_mats"} - set(keys))
        if missing:
            raise MotionValidationError(f"Missing required SOMA arrays: {', '.join(missing)}.")

        try:
            posed_joints = np.asarray(archive["posed_joints"])
            global_rotations = np.asarray(archive["global_rot_mats"])
        except ValueError as exc:
            raise MotionValidationError(
                "Object arrays and pickle-backed fields are not supported."
            ) from exc

        if posed_joints.ndim != 3 or posed_joints.shape[1:] != (SOMA_JOINT_COUNT, 3):
            raise MotionValidationError(
                f"posed_joints must have shape (T, 77, 3); found {posed_joints.shape}."
            )

        frame_count = int(posed_joints.shape[0])
        if frame_count < 1:
            raise MotionValidationError("SOMA motion must contain at least one frame.")
        if frame_count > max_frames:
            raise MotionValidationError(
                f"Motion has {frame_count} frames, exceeding the limit of {max_frames}."
            )

        expected_rot_shape = (frame_count, SOMA_JOINT_COUNT, 3, 3)
        if global_rotations.shape != expected_rot_shape:
            raise MotionValidationError(
                f"global_rot_mats must have shape (T, 77, 3, 3); found {global_rotations.shape}."
            )
        _finite_real_array(posed_joints, name="posed_joints")
        _finite_real_array(global_rotations, name="global_rot_mats")

        for optional_name, expected_tail in (
            ("local_rot_mats", (SOMA_JOINT_COUNT, 3, 3)),
            ("root_positions", (3,)),
            ("smooth_root_pos", (3,)),
            ("global_root_heading", (2,)),
        ):
            if optional_name not in archive.files:
                continue
            try:
                optional_array = np.asarray(archive[optional_name])
            except ValueError as exc:
                raise MotionValidationError(
                    f"{optional_name} is pickle-backed or has object dtype."
                ) from exc
            expected_shape = (frame_count, *expected_tail)
            if optional_array.shape != expected_shape:
                raise MotionValidationError(
                    f"{optional_name} must have shape {expected_shape}; "
                    f"found {optional_array.shape}."
                )
            _finite_real_array(optional_array, name=optional_name)

        contact_channels: int | None = None
        if "foot_contacts" in archive.files:
            try:
                contacts = np.asarray(archive["foot_contacts"])
            except ValueError as exc:
                raise MotionValidationError(
                    "foot_contacts is pickle-backed or has object dtype."
                ) from exc
            if contacts.ndim != 2 or contacts.shape[0] != frame_count:
                raise MotionValidationError(
                    f"foot_contacts must have shape (T, 4) or (T, 6); found {contacts.shape}."
                )
            contact_channels = int(contacts.shape[1])
            if contact_channels not in (4, 6):
                raise MotionValidationError(
                    f"foot_contacts must have 4 or 6 channels; found {contact_channels}."
                )
            if not (np.issubdtype(contacts.dtype, np.bool_) or _is_real_numeric(contacts)):
                raise MotionValidationError("foot_contacts must have a boolean or real dtype.")
            if not np.isfinite(contacts).all():
                raise MotionValidationError("foot_contacts contains NaN or infinity.")

        sample_step = max(1, frame_count // 20)
        sampled_rotations = global_rotations[::sample_step]
        gram = np.swapaxes(sampled_rotations, -1, -2) @ sampled_rotations
        identity = np.eye(3, dtype=sampled_rotations.dtype)
        orthogonality_error = float(np.max(np.abs(gram - identity)))
        determinant_error = float(np.max(np.abs(np.linalg.det(sampled_rotations) - 1.0)))
        if orthogonality_error > 1e-3 or determinant_error > 1e-3:
            warnings.append(
                "Sampled global rotations deviate from SO(3): "
                f"orthogonality={orthogonality_error:.3g}, "
                f"determinant={determinant_error:.3g}."
            )

        fps = _resolve_fps(archive, fps_override, warnings)

    return SomaMotionSummary(
        path=motion_path,
        sha256=_file_digest(motion_path),
        frame_count=frame_count,
        fps=fps,
        duration_seconds=frame_count / fps,
        keys=keys,
        contact_channels=contact_channels,
        warnings=tuple(warnings),
    )


def rotation_x(angle_radians: float) -> NDArray[np.float64]:
    """Return the same elementary X rotation used by the research loader."""

    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=np.float64,
    )


def rotation_z(angle_radians: float) -> NDArray[np.float64]:
    """Return the same elementary Z rotation used by the research loader."""

    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def load_soma_motion(
    path: str | Path,
    *,
    fps_override: float | None = None,
    position_scale: float = 1.0,
    z_up: bool = True,
    position_offset: NDArray[np.float64] | tuple[float, float, float] | None = None,
    rotation_offset: NDArray[np.float64] | None = None,
) -> SomaMotion:
    """Load SOMA arrays and apply the configured source-world conversion.

    With the defaults, this performs the ``zup=True`` conversion:
    positions and global rotations are left-multiplied by ``Rx(+90 deg)``.
    No stance, hip-width, or robot-relative normalization is applied.
    """

    summary = validate_soma_npz(path, fps_override=fps_override)
    try:
        scale = float(position_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionValidationError("position_scale must be a finite real scalar.") from exc
    if not np.isfinite(scale):
        raise MotionValidationError("position_scale must be a finite real scalar.")

    offset = np.zeros(3, dtype=np.float64)
    if position_offset is not None:
        try:
            offset = np.asarray(position_offset, dtype=np.float64).reshape(3)
        except (TypeError, ValueError) as exc:
            raise MotionValidationError("position_offset must contain exactly 3 values.") from exc
    if not np.isfinite(offset).all():
        raise MotionValidationError("position_offset contains NaN or infinity.")

    total_rotation = np.eye(3, dtype=np.float64)
    if rotation_offset is not None:
        try:
            total_rotation = np.asarray(rotation_offset, dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError) as exc:
            raise MotionValidationError("rotation_offset must contain exactly 9 values.") from exc
    if not np.isfinite(total_rotation).all():
        raise MotionValidationError("rotation_offset contains NaN or infinity.")
    if z_up:
        total_rotation = total_rotation @ rotation_x(np.pi / 2.0)

    try:
        with np.load(summary.path, allow_pickle=False) as archive:
            posed_joints = np.asarray(archive["posed_joints"], dtype=np.float64)
            global_rot_mats = np.asarray(archive["global_rot_mats"], dtype=np.float64)
            contacts = (
                np.asarray(archive["foot_contacts"]) if "foot_contacts" in archive.files else None
            )
    except (OSError, ValueError, KeyError) as exc:
        raise MotionValidationError(f"Could not load validated SOMA arrays: {exc}") from exc

    posed_joints = np.einsum("ij,taj->tai", total_rotation, scale * posed_joints) + offset
    global_rot_mats = np.einsum("ij,tajk->taik", total_rotation, global_rot_mats)
    seconds = np.arange(summary.frame_count, dtype=np.float64) / summary.fps

    return SomaMotion(
        summary=summary,
        seconds=seconds,
        posed_joints=posed_joints,
        global_rot_mats=global_rot_mats,
        foot_contacts=contacts,
        world_rotation=total_rotation,
        position_scale=scale,
        position_offset=offset,
        z_up=bool(z_up),
    )
