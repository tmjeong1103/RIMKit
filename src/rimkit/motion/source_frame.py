"""Canonical pelvis-frame conversion for Kimodo SOMA77 motion."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from rimkit.motion.soma import SomaMotion, rotation_x, rotation_z
from rimkit.motion.soma_joints import SOMA77_JOINT_INDEX

SOMA_PELVIS_LOCAL_ALIGNMENT: Final[NDArray[np.float64]] = rotation_x(-np.pi / 2.0) @ rotation_z(
    -np.pi / 2.0
)
SOMA_PELVIS_LOCAL_ALIGNMENT.setflags(write=False)


def canonical_soma_pelvis_rotations(
    motion: SomaMotion,
) -> NDArray[np.float64]:
    """Return the exact fixed SOMA pelvis-local conversion used by DMR.

    The motion loader has already performed the optional world-frame Z-up
    conversion. This function therefore only right-multiplies the SOMA
    Hips-local alignment; it does not rotate the world frame a second time.
    """

    hips_rotations = motion.global_rot_mats[:, SOMA77_JOINT_INDEX["Hips"]]
    canonical = np.asarray(np.matmul(hips_rotations, SOMA_PELVIS_LOCAL_ALIGNMENT), dtype=np.float64)
    canonical.setflags(write=False)
    return canonical
