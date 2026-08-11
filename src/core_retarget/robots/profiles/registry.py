"""Registry boundary for robot-specific DMR profiles."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Literal

from core_retarget.exceptions import ConfigurationError
from core_retarget.robots.profiles.adam import ADAM_DMR_PROFILE
from core_retarget.robots.profiles.apollo import APOLLO_DMR_PROFILE
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE
from core_retarget.robots.profiles.h1 import H1_DMR_PROFILE
from core_retarget.robots.profiles.h2 import H2_DMR_PROFILE
from core_retarget.robots.profiles.k1 import K1_DMR_PROFILE
from core_retarget.robots.profiles.n1 import N1_DMR_PROFILE
from core_retarget.robots.profiles.oli import OLI_DMR_PROFILE
from core_retarget.robots.profiles.pm01 import PM01_DMR_PROFILE
from core_retarget.robots.profiles.r1 import R1_DMR_PROFILE
from core_retarget.robots.profiles.schema import DmrProfile
from core_retarget.robots.profiles.t1 import T1_DMR_PROFILE
from core_retarget.robots.registry import get_robot

DMR_PROFILES = MappingProxyType(
    {
        "g1": G1_DMR_PROFILE,
        "h1": H1_DMR_PROFILE,
        "h2": H2_DMR_PROFILE,
        "r1": R1_DMR_PROFILE,
        "k1": K1_DMR_PROFILE,
        "apollo": APOLLO_DMR_PROFILE,
        "oli": OLI_DMR_PROFILE,
        "n1": N1_DMR_PROFILE,
        "adam": ADAM_DMR_PROFILE,
        "t1": T1_DMR_PROFILE,
        "pm01": PM01_DMR_PROFILE,
    }
)


def get_dmr_profile(
    robot_id: str,
    *,
    source_provider: Literal["kimodo", "gem-x"] = "kimodo",
) -> DmrProfile:
    """Return the configured DMR profile for a supported robot.

    The profile registry covers the same public robot IDs as the asset
    registry.  Keeping the lookup boundary explicit prevents an asset-only
    entry from silently becoming runnable if those registries diverge later.
    """

    robot = get_robot(robot_id)
    try:
        profile = DMR_PROFILES[robot.robot_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"No verified DMR profile is registered for robot {robot.robot_id!r}."
        ) from exc
    if source_provider == "kimodo":
        return profile
    if source_provider != "gem-x":
        raise ConfigurationError(f"Unsupported source provider {source_provider!r}.")

    profile = replace(
        profile,
        orientation_smoothing_mode="quaternion_continuous",
        dmr_initial_nullspace_gain=1.0,
    )
    if robot.robot_id == "h1":
        return replace(
            profile,
            ankle_contact_flatten_strength=1.0,
            ankle_contact_flatten_smooth_time=0.10,
        )
    if robot.robot_id != "k1":
        return replace(
            profile,
            left_ankle_orientation_joi_key="lsole",
            right_ankle_orientation_joi_key="rsole",
            ankle_contact_flatten_strength=1.0,
            ankle_contact_flatten_smooth_time=0.06,
        )
    return profile


__all__ = ["DMR_PROFILES", "get_dmr_profile"]
