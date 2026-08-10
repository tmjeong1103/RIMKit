"""Immutable robot-neutral tuning for affine root adjustment (ARA)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from core_retarget.exceptions import ConfigurationError
from core_retarget.robots.registry import get_robot


@dataclass(frozen=True, slots=True)
class AraProfile:
    """Numerical constants used by the shared Stage 5 ARA optimization.

    The research notebooks use the same values.  Keeping one immutable
    profile instance makes that robot-neutral contract explicit while the
    registry still prevents an asset-only robot from becoming runnable by
    accident.
    """

    slip_weight: float = 100.0
    floor_lock_weight: float = 300.0
    scale_regularization_weight: float = 0.1
    shift_regularization_weight: float = 1e-3
    scale_xy_bound: float = 0.40
    shift_xy_bound: float = 0.40
    ground_target_z: float = 0.0
    toe_floor_offset: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = (
            "slip_weight",
            "floor_lock_weight",
            "scale_regularization_weight",
            "shift_regularization_weight",
            "scale_xy_bound",
            "shift_xy_bound",
        )
        for field_name in nonnegative:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, value)
        for field_name in ("ground_target_z", "toe_floor_offset"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)


ROBOT_NEUTRAL_ARA_PROFILE = AraProfile()

ARA_PROFILES = MappingProxyType(
    {
        robot_id: ROBOT_NEUTRAL_ARA_PROFILE
        for robot_id in (
            "g1",
            "h1",
            "h2",
            "r1",
            "k1",
            "apollo",
            "oli",
            "n1",
            "adam",
            "t1",
            "pm01",
        )
    }
)


def get_ara_profile(robot_id: str) -> AraProfile:
    """Return the verified shared ARA profile for one supported robot."""

    robot = get_robot(robot_id)
    try:
        return ARA_PROFILES[robot.robot_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"No verified ARA profile is registered for robot {robot.robot_id!r}."
        ) from exc


__all__ = [
    "ARA_PROFILES",
    "ROBOT_NEUTRAL_ARA_PROFILE",
    "AraProfile",
    "get_ara_profile",
]
