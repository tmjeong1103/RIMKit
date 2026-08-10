"""Production-safe, versioned robot-motion NPZ export.

The public format stores MuJoCo qpos directly and never serializes Python
objects or pickle payloads.  A completed archive is validated with
``allow_pickle=False`` before an atomic same-directory replace publishes it.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget._version import __version__
from core_retarget.assets import root_path
from core_retarget.exceptions import ArtifactError, MotionValidationError
from core_retarget.mujoco.model import MujocoModel
from core_retarget.robots.registry import get_robot

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ROBOT_MOTION_FORMAT = "core-robot-motion-v1"
ROBOT_MOTION_SCHEMA_VERSION = 1
CONTACT_LABEL_NAMES = ("left_foot", "right_foot", "left_hand", "right_hand")
ROOT_QPOS_NAMES = (
    "root_x",
    "root_y",
    "root_z",
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
)
QPOS_LAYOUT = "root_xyz_quat_wxyz_then_model_joint_qpos"


@dataclass(frozen=True, slots=True)
class RobotMotionArtifact:
    """Identity and checksum of one atomically published motion archive."""

    path: Path
    sha256: str
    robot_id: str
    frame_count: int
    qpos_dim: int
    fps: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    invalid_character = any(character not in "0123456789abcdef" for character in normalized)
    if len(normalized) != 64 or invalid_character:
        raise MotionValidationError(f"{name} must be a 64-character hexadecimal SHA-256.")
    return normalized


def _coerce_binary(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.hasobject or raw.dtype.kind in "SUc":
        raise MotionValidationError(f"{name} must contain boolean or binary numeric values.")
    if raw.dtype.kind in "fiu":
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any((numeric != 0.0) & (numeric != 1.0)):
            raise MotionValidationError(f"{name} must contain only zero or one.")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def _contact_segments(label: BoolArray) -> IntArray:
    padded = np.concatenate(
        [np.asarray([False]), np.asarray(label, dtype=np.bool_), np.asarray([False])]
    ).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return np.asarray(np.column_stack([starts, ends]).reshape(-1, 2), dtype=np.int64)


@lru_cache(maxsize=16)
def _robot_joint_names(robot_id: str) -> tuple[str, ...]:
    model = MujocoModel.from_robot(robot_id)
    return model.qpos_joint_names


@lru_cache(maxsize=16)
def _scene_sha256(robot_id: str) -> str:
    robot = get_robot(robot_id)
    return _sha256(root_path() / robot.scene_relpath)


def build_robot_motion_arrays(
    *,
    robot_id: str,
    qpos: ArrayLike,
    seconds: ArrayLike,
    fps: float,
    contact_labels: ArrayLike,
    contact_confidence: ArrayLike,
    contact_availability: ArrayLike,
    flight_labels: ArrayLike | None = None,
    source_motion_sha256: str,
    contact_source: str = "unspecified",
    hand_contact_source: str = "unspecified",
) -> dict[str, NDArray[np.generic]]:
    """Validate ArrayLike inputs and build an object-free production payload."""

    robot = get_robot(robot_id)
    time_values = np.array(seconds, dtype=np.float64, copy=True, order="C").reshape(-1)
    trajectory = np.array(qpos, dtype=np.float64, copy=True, order="C")
    frame_count = len(time_values)
    if frame_count == 0:
        raise MotionValidationError("Robot-motion export requires at least one frame.")
    if trajectory.shape != (frame_count, robot.expected_nq):
        raise MotionValidationError(
            "Robot-motion qpos must have shape "
            f"({frame_count}, {robot.expected_nq}); found {trajectory.shape}."
        )
    if not np.isfinite(time_values).all() or not np.isfinite(trajectory).all():
        raise MotionValidationError("Robot-motion qpos and seconds must be finite.")
    if frame_count > 1 and np.any(np.diff(time_values) <= 0.0):
        raise MotionValidationError("Robot-motion seconds must be strictly increasing.")

    try:
        fps_value = float(fps)
    except (TypeError, ValueError, OverflowError) as error:
        raise MotionValidationError("Robot-motion FPS must be finite and positive.") from error
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise MotionValidationError("Robot-motion FPS must be finite and positive.")
    if frame_count > 1 and not np.allclose(
        np.diff(time_values),
        1.0 / fps_value,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise MotionValidationError("Robot-motion seconds must be uniformly sampled at FPS.")

    quaternion_norm = np.linalg.norm(trajectory[:, 3:7], axis=1)
    if np.any(quaternion_norm <= 1e-12) or np.any(np.abs(quaternion_norm - 1.0) > 1e-5):
        raise MotionValidationError(
            "Robot-motion root quaternions must be nonzero and unit normalized."
        )

    labels = _coerce_binary(contact_labels, name="Robot-motion contact labels")
    confidence = np.array(contact_confidence, dtype=np.float64, copy=True, order="C")
    availability = _coerce_binary(
        contact_availability,
        name="Robot-motion contact availability",
    ).reshape(-1)
    if labels.shape != (frame_count, 4):
        raise MotionValidationError(
            f"Robot-motion contact labels must have shape ({frame_count}, 4)."
        )
    if confidence.shape != (frame_count, 4):
        raise MotionValidationError(
            f"Robot-motion contact confidence must have shape ({frame_count}, 4)."
        )
    if availability.shape != (4,):
        raise MotionValidationError("Robot-motion contact availability must have shape (4,).")
    if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise MotionValidationError(
            "Robot-motion contact confidence must be finite and lie in [0, 1]."
        )
    if np.any(labels[:, ~availability]) or np.any(confidence[:, ~availability] != 0.0):
        raise MotionValidationError(
            "Unavailable robot-motion contact channels must be false with zero confidence."
        )

    flight = (
        ~(labels[:, 0] | labels[:, 1])
        if flight_labels is None
        else _coerce_binary(flight_labels, name="Robot-motion flight labels").reshape(-1)
    )
    if flight.shape != (frame_count,):
        raise MotionValidationError(f"Robot-motion flight labels must have shape ({frame_count},).")

    source_hash = _validate_sha256(
        source_motion_sha256,
        name="source_motion_sha256",
    )
    if not str(contact_source).strip() or not str(hand_contact_source).strip():
        raise MotionValidationError("Robot-motion contact source labels must not be empty.")

    joint_names = _robot_joint_names(robot.robot_id)
    if len(joint_names) != robot.expected_nq - 7:
        raise ArtifactError(f"Robot {robot.robot_id!r} joint layout does not match qpos dimension.")

    arrays: dict[str, NDArray[np.generic]] = {
        "schema_version": np.asarray(ROBOT_MOTION_SCHEMA_VERSION, dtype=np.int32),
        "format": np.asarray(ROBOT_MOTION_FORMAT),
        "core_version": np.asarray(__version__),
        "robot_id": np.asarray(robot.robot_id),
        "model_sha256": np.asarray(robot.model_sha256),
        "scene_sha256": np.asarray(_scene_sha256(robot.robot_id)),
        "source_motion_sha256": np.asarray(source_hash),
        "fps": np.asarray(fps_value, dtype=np.float64),
        "timestamps_s": time_values,
        "qpos": trajectory,
        "qpos_layout": np.asarray(QPOS_LAYOUT),
        "root_qpos_names": np.asarray(ROOT_QPOS_NAMES),
        "joint_names": np.asarray(joint_names),
        "contact_label_names": np.asarray(CONTACT_LABEL_NAMES),
        "contact_labels": labels,
        "contact_confidence": confidence,
        "contact_availability": availability,
        "flight_labels": flight,
        "left_contact_segments": _contact_segments(labels[:, 0]),
        "right_contact_segments": _contact_segments(labels[:, 1]),
        "contact_source": np.asarray(str(contact_source).strip()),
        "hand_contact_source": np.asarray(str(hand_contact_source).strip()),
    }
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ArtifactError(f"Object arrays are forbidden in robot-motion output: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ArtifactError(f"Non-finite values are forbidden in robot-motion output: {name}")
    return arrays


def _validate_written_archive(
    path: Path,
    expected: dict[str, NDArray[np.generic]],
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != tuple(expected):
                raise ArtifactError("Robot-motion archive member order or set changed on write.")
            for name, expected_value in expected.items():
                actual = archive[name]
                expected_array = np.asarray(expected_value)
                if actual.dtype.hasobject:
                    raise ArtifactError(
                        f"Object array entered robot-motion archive member {name!r}."
                    )
                if actual.dtype != expected_array.dtype or actual.shape != expected_array.shape:
                    raise ArtifactError(
                        f"Robot-motion archive member {name!r} changed dtype or shape."
                    )
                if not np.array_equal(actual, expected_array):
                    raise ArtifactError(
                        f"Robot-motion archive member {name!r} changed during serialization."
                    )
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Could not validate robot-motion archive {path}: {error}") from error


def write_robot_motion_npz(
    path: str | Path,
    *,
    robot_id: str,
    qpos: ArrayLike,
    seconds: ArrayLike,
    fps: float,
    contact_labels: ArrayLike,
    contact_confidence: ArrayLike,
    contact_availability: ArrayLike,
    flight_labels: ArrayLike | None = None,
    source_motion_sha256: str,
    contact_source: str = "unspecified",
    hand_contact_source: str = "unspecified",
    overwrite: bool = False,
) -> RobotMotionArtifact:
    """Atomically write one validated, pickle-free robot-motion NPZ."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ArtifactError("Robot-motion output path must use the .npz extension.")
    if destination.exists() and not overwrite:
        raise ArtifactError(f"Robot-motion output already exists: {destination}")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ArtifactError(f"Robot-motion output parent is not a directory: {destination.parent}")

    arrays = build_robot_motion_arrays(
        robot_id=robot_id,
        qpos=qpos,
        seconds=seconds,
        fps=fps,
        contact_labels=contact_labels,
        contact_confidence=contact_confidence,
        contact_availability=contact_availability,
        flight_labels=flight_labels,
        source_motion_sha256=source_motion_sha256,
        contact_source=contact_source,
        hand_contact_source=hand_contact_source,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.npz",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        _validate_written_archive(temporary, arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Robot-motion output appeared during write: {destination}")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    robot = get_robot(robot_id)
    return RobotMotionArtifact(
        path=destination,
        sha256=_sha256(destination),
        robot_id=robot.robot_id,
        frame_count=int(np.asarray(arrays["qpos"]).shape[0]),
        qpos_dim=int(np.asarray(arrays["qpos"]).shape[1]),
        fps=float(np.asarray(arrays["fps"])),
    )


__all__ = [
    "CONTACT_LABEL_NAMES",
    "QPOS_LAYOUT",
    "ROBOT_MOTION_FORMAT",
    "ROBOT_MOTION_SCHEMA_VERSION",
    "ROOT_QPOS_NAMES",
    "RobotMotionArtifact",
    "build_robot_motion_arrays",
    "write_robot_motion_npz",
]
