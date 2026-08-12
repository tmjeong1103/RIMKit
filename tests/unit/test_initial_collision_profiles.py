from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from core_retarget.robots.profiles import (
    ADAM_INITIAL_COLLISION_PROFILE,
    APOLLO_INITIAL_COLLISION_PROFILE,
    G1_INITIAL_COLLISION_PROFILE,
    H1_INITIAL_COLLISION_PROFILE,
    H2_INITIAL_COLLISION_PROFILE,
    INITIAL_COLLISION_PROFILES,
    K1_INITIAL_COLLISION_PROFILE,
    N1_INITIAL_COLLISION_PROFILE,
    OLI_INITIAL_COLLISION_PROFILE,
    PM01_INITIAL_COLLISION_PROFILE,
    R1_INITIAL_COLLISION_PROFILE,
    T1_INITIAL_COLLISION_PROFILE,
    get_initial_collision_profile,
)
from core_retarget.robots.profiles.schema import InitialCollisionProfile

PROFILES = {
    "g1": G1_INITIAL_COLLISION_PROFILE,
    "h1": H1_INITIAL_COLLISION_PROFILE,
    "h2": H2_INITIAL_COLLISION_PROFILE,
    "r1": R1_INITIAL_COLLISION_PROFILE,
    "k1": K1_INITIAL_COLLISION_PROFILE,
    "apollo": APOLLO_INITIAL_COLLISION_PROFILE,
    "oli": OLI_INITIAL_COLLISION_PROFILE,
    "n1": N1_INITIAL_COLLISION_PROFILE,
    "adam": ADAM_INITIAL_COLLISION_PROFILE,
    "t1": T1_INITIAL_COLLISION_PROFILE,
    "pm01": PM01_INITIAL_COLLISION_PROFILE,
}
QPOS_DIMS = {
    "g1": 36,
    "h1": 27,
    "h2": 38,
    "r1": 36,
    "k1": 30,
    "apollo": 39,
    "oli": 68,
    "n1": 30,
    "adam": 32,
    "t1": 30,
    "pm01": 31,
}
ROOT_BODIES = {
    "g1": "pelvis",
    "h1": "pelvis",
    "h2": "pelvis",
    "r1": "pelvis",
    "k1": "pelvis",
    "apollo": "base_link",
    "oli": "base_link",
    "n1": "base_link",
    "adam": "pelvis",
    "t1": "Trunk",
    "pm01": "LINK_BASE",
}


@pytest.mark.parametrize("robot_id", tuple(PROFILES))
def test_initial_collision_profile_registry_is_case_insensitive(robot_id: str) -> None:
    expected = PROFILES[robot_id]

    assert get_initial_collision_profile(f" {robot_id.upper()} ") is expected
    assert INITIAL_COLLISION_PROFILES[robot_id] is expected


def test_initial_collision_registry_contains_all_supported_robots() -> None:
    assert tuple(INITIAL_COLLISION_PROFILES) == tuple(PROFILES)
    assert set(INITIAL_COLLISION_PROFILES) == set(PROFILES)


@pytest.mark.parametrize(("robot_id", "qpos_dim"), tuple(QPOS_DIMS.items()))
def test_initial_collision_profiles_match_the_public_stage3_constants(
    robot_id: str,
    qpos_dim: int,
) -> None:
    profile = get_initial_collision_profile(robot_id)

    assert profile.robot_id == robot_id
    assert profile.qpos_dim == qpos_dim
    assert profile.initial_margin == 0.03
    assert profile.correction_gain == 0.5
    assert profile.ticks_per_pass == 24
    assert profile.correction_length_cap == 0.03
    assert profile.outer_passes == 2
    assert profile.margin_scale == 1.4
    assert profile.margin_cap == 0.03
    assert profile.query_limit == 32
    assert profile.target_limit == 16
    assert profile.ancestor_skip_depth == 2
    assert profile.root_body_name == ROOT_BODIES[robot_id]
    assert profile.movable_joint_tokens == ("shoulder", "elbow", "wrist", "arm")
    assert profile.movable_body_tokens == (
        "shoulder",
        "elbow",
        "wrist",
        "hand",
        "palm",
    )
    assert profile.preserve_orientation
    assert profile.orientation_weight == 0.03
    assert profile.orientation_axis_length == 0.08
    assert profile.smooth_each_pass
    assert profile.final_pass_without_smoothing
    assert profile.final_pass_margin == 0.002
    assert profile.smooth_jerk_weight == 1e-5
    assert profile.smooth_tracking_norm == 1

    assert profile.solver.max_iterations == 10
    assert profile.solver.revolute_step == 0.5
    assert profile.solver.revolute_update_limit == pytest.approx(math.radians(5.0))
    assert profile.solver.damping == 1e-2
    assert profile.solver.joint_limit_probe == pytest.approx(math.radians(3.0))


def test_legacy_robot_profiles_differ_only_in_identity_and_qpos_dimension() -> None:
    for robot_id in ("g1", "h1", "h2", "r1", "k1"):
        profile = PROFILES[robot_id]
        for field in fields(InitialCollisionProfile):
            if field.name in {"robot_id", "qpos_dim"}:
                continue
            assert getattr(profile, field.name) == getattr(
                G1_INITIAL_COLLISION_PROFILE,
                field.name,
            ), (robot_id, field.name)


def test_new_robot_collision_prefixes_match_research_profiles() -> None:
    assert APOLLO_INITIAL_COLLISION_PROFILE.movable_body_prefixes == ("l_", "r_")
    assert OLI_INITIAL_COLLISION_PROFILE.movable_body_prefixes == ("left_", "right_")
    assert N1_INITIAL_COLLISION_PROFILE.movable_body_prefixes == ("left_", "right_")
    assert ADAM_INITIAL_COLLISION_PROFILE.movable_body_prefixes == (
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
    )
    assert T1_INITIAL_COLLISION_PROFILE.movable_body_prefixes == (
        "H",
        "AL",
        "AR",
        "Waist",
        "Hip_",
        "Shank_",
        "Ankle_",
        "left_",
        "right_",
    )
    assert PM01_INITIAL_COLLISION_PROFILE.movable_body_prefixes == (
        "LINK_HIP_",
        "LINK_KNEE_",
        "LINK_ANKLE_",
        "LINK_FOOT_",
        "LINK_SHOULDER_",
        "LINK_ELBOW_",
        "LINK_WRIST_",
        "LINK_HAND_",
    )


def test_initial_collision_profiles_and_registry_are_deeply_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        G1_INITIAL_COLLISION_PROFILE.qpos_dim = 37  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        G1_INITIAL_COLLISION_PROFILE.solver.max_iterations = 11  # type: ignore[misc]
    with pytest.raises(TypeError):
        INITIAL_COLLISION_PROFILES["g1"] = H1_INITIAL_COLLISION_PROFILE  # type: ignore[index]
    with pytest.raises(TypeError):
        G1_INITIAL_COLLISION_PROFILE.movable_joint_tokens[0] = "changed"  # type: ignore[index]

    external_tokens = ["shoulder", "elbow"]
    copied = replace(
        G1_INITIAL_COLLISION_PROFILE,
        movable_joint_tokens=external_tokens,  # type: ignore[arg-type]
    )
    external_tokens.append("changed")
    assert copied.movable_joint_tokens == ("shoulder", "elbow")
