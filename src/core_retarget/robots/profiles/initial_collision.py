"""Profiles for the first post-DMR signed-distance collision pass."""

from __future__ import annotations

from dataclasses import replace
from math import radians
from types import MappingProxyType

from core_retarget.exceptions import ConfigurationError
from core_retarget.robots.profiles.schema import IkSolverProfile, InitialCollisionProfile
from core_retarget.robots.registry import get_robot

_SOLVER = IkSolverProfile(
    max_iterations=10,
    revolute_step=0.5,
    revolute_update_limit=radians(5.0),
    damping=1e-2,
    joint_limit_probe=radians(3.0),
)

G1_INITIAL_COLLISION_PROFILE = InitialCollisionProfile(
    robot_id="g1",
    qpos_dim=36,
    solver=_SOLVER,
)
H1_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="h1",
    qpos_dim=27,
)
H2_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="h2",
    qpos_dim=38,
)
R1_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="r1",
    qpos_dim=36,
)
K1_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="k1",
    qpos_dim=30,
)
APOLLO_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="apollo",
    qpos_dim=39,
    root_body_name="base_link",
    movable_body_prefixes=("l_", "r_"),
)
OLI_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="oli",
    qpos_dim=68,
    root_body_name="base_link",
)
N1_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="n1",
    qpos_dim=30,
    root_body_name="base_link",
)
ADAM_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="adam",
    qpos_dim=32,
    root_body_name="pelvis",
    movable_body_prefixes=(
        "hipPitch",
        "hipRoll",
        "thigh",
        "shin",
        "anklePitch",
        "toe",
        "shoulderPitch",
        "shoulderRoll",
        "shoulderYaw",
        "elbow",
        "wristYaw",
    ),
)
T1_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="t1",
    qpos_dim=30,
    root_body_name="Trunk",
    movable_body_prefixes=(
        "H",
        "AL",
        "AR",
        "Waist",
        "Hip_",
        "Shank_",
        "Ankle_",
        "left_",
        "right_",
    ),
)
PM01_INITIAL_COLLISION_PROFILE = replace(
    G1_INITIAL_COLLISION_PROFILE,
    robot_id="pm01",
    qpos_dim=31,
    root_body_name="LINK_BASE",
    movable_body_prefixes=(
        "LINK_HIP_",
        "LINK_KNEE_",
        "LINK_ANKLE_",
        "LINK_FOOT_",
        "LINK_SHOULDER_",
        "LINK_ELBOW_",
        "LINK_WRIST_",
        "LINK_HAND_",
    ),
)

INITIAL_COLLISION_PROFILES = MappingProxyType(
    {
        profile.robot_id: profile
        for profile in (
            G1_INITIAL_COLLISION_PROFILE,
            H1_INITIAL_COLLISION_PROFILE,
            H2_INITIAL_COLLISION_PROFILE,
            R1_INITIAL_COLLISION_PROFILE,
            K1_INITIAL_COLLISION_PROFILE,
            APOLLO_INITIAL_COLLISION_PROFILE,
            OLI_INITIAL_COLLISION_PROFILE,
            N1_INITIAL_COLLISION_PROFILE,
            ADAM_INITIAL_COLLISION_PROFILE,
            T1_INITIAL_COLLISION_PROFILE,
            PM01_INITIAL_COLLISION_PROFILE,
        )
    }
)


def get_initial_collision_profile(robot_id: str) -> InitialCollisionProfile:
    """Return the Stage 3 settings for one supported robot."""

    robot = get_robot(robot_id)
    try:
        return INITIAL_COLLISION_PROFILES[robot.robot_id]
    except KeyError as exc:
        raise ConfigurationError(
            f"No initial-collision profile is registered for robot {robot.robot_id!r}."
        ) from exc


__all__ = [
    "ADAM_INITIAL_COLLISION_PROFILE",
    "APOLLO_INITIAL_COLLISION_PROFILE",
    "G1_INITIAL_COLLISION_PROFILE",
    "H1_INITIAL_COLLISION_PROFILE",
    "H2_INITIAL_COLLISION_PROFILE",
    "INITIAL_COLLISION_PROFILES",
    "K1_INITIAL_COLLISION_PROFILE",
    "N1_INITIAL_COLLISION_PROFILE",
    "OLI_INITIAL_COLLISION_PROFILE",
    "PM01_INITIAL_COLLISION_PROFILE",
    "R1_INITIAL_COLLISION_PROFILE",
    "T1_INITIAL_COLLISION_PROFILE",
    "get_initial_collision_profile",
]
