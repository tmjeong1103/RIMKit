"""SOMA77 topology and source joints-of-interest used by DMR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rimkit.exceptions import MotionValidationError
from rimkit.motion.soma import SOMA_JOINT_COUNT, SomaMotion

SOMA77_JOINT_PARENTS: Final[tuple[tuple[str, str | None], ...]] = (
    ("Hips", None),
    ("Spine1", "Hips"),
    ("Spine2", "Spine1"),
    ("Chest", "Spine2"),
    ("Neck1", "Chest"),
    ("Neck2", "Neck1"),
    ("Head", "Neck2"),
    ("HeadEnd", "Head"),
    ("Jaw", "Head"),
    ("LeftEye", "Head"),
    ("RightEye", "Head"),
    ("LeftShoulder", "Chest"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"),
    ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"),
    ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"),
    ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"),
    ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"),
    ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"),
    ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"),
    ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"),
    ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"),
    ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"),
    ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"),
    ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"),
    ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"),
    ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"),
    ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"),
    ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"),
    ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"),
    ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"),
    ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"),
    ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"),
    ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"),
    ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"),
    ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"),
    ("RightToeEnd", "RightToeBase"),
)

if len(SOMA77_JOINT_PARENTS) != SOMA_JOINT_COUNT:
    raise RuntimeError("SOMA77 topology must contain exactly 77 joints.")

SOMA77_JOINT_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in SOMA77_JOINT_PARENTS)
SOMA77_JOINT_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(SOMA77_JOINT_NAMES)}
)

SOMA_JOI_NAMES: Final[tuple[str, ...]] = (
    "base",
    "spine",
    "rs",
    "re",
    "rw",
    "rh",
    "ls",
    "le",
    "lw",
    "lh",
    "neck",
    "head",
    "rp",
    "rk",
    "ra",
    "rf",
    "rtoe",
    "lp",
    "lk",
    "la",
    "lf",
    "ltoe",
)
SOMA_JOI_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(SOMA_JOI_NAMES)}
)

_JOI_SOURCE_JOINTS: Final[tuple[tuple[str, str], ...]] = (
    ("base", "Hips"),
    ("spine", "Spine2"),
    ("rs", "RightArm"),
    ("re", "RightForeArm"),
    ("rw", "RightHand"),
    ("rh", "RightHand"),
    ("ls", "LeftArm"),
    ("le", "LeftForeArm"),
    ("lw", "LeftHand"),
    ("lh", "LeftHand"),
    ("neck", "Neck1"),
    ("head", "Head"),
    ("rp", "RightLeg"),
    ("rk", "RightShin"),
    ("ra", "RightFoot"),
    ("rf", "RightFoot"),
    ("rtoe", "RightToeBase"),
    ("lp", "LeftLeg"),
    ("lk", "LeftShin"),
    ("la", "LeftFoot"),
    ("lf", "LeftFoot"),
    ("ltoe", "LeftToeBase"),
)


@dataclass(frozen=True)
class SomaJoiTrajectory:
    """Frame-wise global transforms for the 22 source JOIs used by CoRe."""

    seconds: NDArray[np.float64]
    transforms: NDArray[np.float64]

    def __post_init__(self) -> None:
        seconds = np.array(self.seconds, dtype=np.float64, copy=True, order="C")
        transforms = np.array(self.transforms, dtype=np.float64, copy=True, order="C")
        expected_shape = (seconds.shape[0], len(SOMA_JOI_NAMES), 4, 4)
        if seconds.ndim != 1:
            raise MotionValidationError("JOI seconds must be a one-dimensional array.")
        if transforms.shape != expected_shape:
            raise MotionValidationError(
                f"JOI transforms must have shape {expected_shape}; found {transforms.shape}."
            )
        if not np.isfinite(seconds).all() or not np.isfinite(transforms).all():
            raise MotionValidationError("JOI trajectory contains NaN or infinity.")
        seconds.setflags(write=False)
        transforms.setflags(write=False)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "transforms", transforms)

    @property
    def names(self) -> tuple[str, ...]:
        """Stable JOI ordering used by the transform array."""

        return SOMA_JOI_NAMES

    @property
    def frame_count(self) -> int:
        """Number of trajectory frames."""

        return int(self.transforms.shape[0])

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        """Return all frame transforms for one JOI name."""

        try:
            index = SOMA_JOI_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"Unknown SOMA JOI {name!r}.") from exc
        return self.transforms[:, index]

    def positions(self, name: str) -> NDArray[np.float64]:
        """Return all frame positions for one JOI name."""

        return self[name][:, :3, 3]

    def rotations(self, name: str) -> NDArray[np.float64]:
        """Return all frame rotations for one JOI name."""

        return self[name][:, :3, :3]


def extract_soma_joi(
    motion: SomaMotion,
    *,
    base_between_pelvis: bool = True,
) -> SomaJoiTrajectory:
    """Extract the global transforms of the SOMA joint-of-interest set."""

    frame_count = motion.frame_count
    transforms = np.zeros((frame_count, len(SOMA_JOI_NAMES), 4, 4), dtype=np.float64)
    transforms[:, :, 3, 3] = 1.0

    for joi_name, joint_name in _JOI_SOURCE_JOINTS:
        joi_index = SOMA_JOI_INDEX[joi_name]
        joint_index = SOMA77_JOINT_INDEX[joint_name]
        transforms[:, joi_index, :3, :3] = motion.global_rot_mats[:, joint_index]
        transforms[:, joi_index, :3, 3] = motion.posed_joints[:, joint_index]

    if base_between_pelvis:
        right_hip = motion.posed_joints[:, SOMA77_JOINT_INDEX["RightLeg"]]
        left_hip = motion.posed_joints[:, SOMA77_JOINT_INDEX["LeftLeg"]]
        transforms[:, SOMA_JOI_INDEX["base"], :3, 3] = 0.5 * (right_hip + left_hip)

    right_clavicle = motion.posed_joints[:, SOMA77_JOINT_INDEX["RightShoulder"]]
    left_clavicle = motion.posed_joints[:, SOMA77_JOINT_INDEX["LeftShoulder"]]
    transforms[:, SOMA_JOI_INDEX["neck"], :3, 3] = 0.5 * (right_clavicle + left_clavicle)

    return SomaJoiTrajectory(seconds=motion.seconds, transforms=transforms)
