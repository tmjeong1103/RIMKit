from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import isinf

import pytest

from core_retarget.mujoco import MujocoModel
from core_retarget.robots.profiles.fpa import (
    FPA_PROFILES,
    FpaProfile,
    get_fpa_profile,
)
from core_retarget.stages.fpa import _fpa_joint_groups


def test_fpa_registry_covers_all_supported_robots_and_is_immutable() -> None:
    assert tuple(FPA_PROFILES) == (
        "k1",
        "h1",
        "g1",
        "h2",
        "r1",
        "apollo",
        "oli",
        "n1",
        "adam",
        "t1",
        "pm01",
    )
    with pytest.raises(TypeError):
        FPA_PROFILES["other"] = FpaProfile(robot_id="other")  # type: ignore[index]


@pytest.mark.parametrize("robot_id", ("k1", "h1"))
def test_k1_family_uses_the_unmodified_research_fpa_defaults(robot_id: str) -> None:
    profile = get_fpa_profile(robot_id.upper())
    assert profile.robot_id == robot_id
    assert profile.joint_correction_median_window == 1
    assert profile.joint_correction_smooth_time == 0.0
    assert isinf(profile.swing_outlier_threshold)
    assert profile.post_ground_dual_support_lower_max == 0.0
    assert profile.post_ground_dual_recovery_passes == 0


@pytest.mark.parametrize(
    "robot_id",
    ("g1", "h2", "r1", "apollo", "oli", "n1", "adam", "t1", "pm01"),
)
def test_g1_family_uses_the_verified_dual_support_profile(robot_id: str) -> None:
    profile = get_fpa_profile(robot_id)
    assert profile.robot_id == robot_id
    assert profile.joint_correction_median_window == 3
    assert profile.joint_correction_smooth_time == 0.06
    assert profile.post_ground_dual_support_lower_max == 0.012
    assert profile.post_ground_dual_recovery_passes == 2


@pytest.mark.parametrize(
    ("robot_id", "left_tokens", "right_tokens"),
    (
        ("apollo", ("l_",), ("r_",)),
        ("adam", ("_Left",), ("_Right",)),
        ("t1", ("Left_",), ("Right_",)),
        ("pm01", ("_L",), ("_R",)),
    ),
)
def test_new_fpa_profiles_resolve_model_specific_leg_names(
    robot_id: str,
    left_tokens: tuple[str, ...],
    right_tokens: tuple[str, ...],
) -> None:
    profile = get_fpa_profile(robot_id)
    assert profile.left_joint_tokens == left_tokens
    assert profile.right_joint_tokens == right_tokens


@pytest.mark.parametrize(
    ("robot_id", "excluded", "lift_max"),
    (("t1", ("Waist",), 0.040), ("pm01", ("J12_WAIST_YAW",), 0.010)),
)
def test_waist_exclusion_and_micro_lift_match_research_profiles(
    robot_id: str,
    excluded: tuple[str, ...],
    lift_max: float,
) -> None:
    profile = get_fpa_profile(robot_id)
    assert profile.excluded_joint_tokens == excluded
    assert profile.post_ground_micro_lift_max == lift_max
    assert profile.post_ground_micro_lift_speed == 0.20
    assert not profile.post_ground_micro_lift_include_swing_feet


@pytest.mark.mujoco
@pytest.mark.parametrize("robot_id", ("apollo", "oli", "n1", "adam", "t1", "pm01"))
def test_new_fpa_profiles_resolve_exactly_six_recovery_joints_per_leg(
    robot_id: str,
) -> None:
    groups = _fpa_joint_groups(MujocoModel.from_robot(robot_id), get_fpa_profile(robot_id))
    assert len(groups.recovery) == 12
    assert len(groups.left_recovery) == 6
    assert len(groups.right_recovery) == 6
    assert not set(groups.left_recovery).intersection(groups.right_recovery)


def test_fpa_profiles_are_frozen_and_validate_active_contracts() -> None:
    profile = get_fpa_profile("g1")
    with pytest.raises(FrozenInstanceError):
        profile.robot_id = "h2"  # type: ignore[misc]
    with pytest.raises(ValueError, match="quantile"):
        replace(profile, sole_clearance_quantile=1.1)
    with pytest.raises(ValueError, match="safety_iterations"):
        replace(profile, post_ground_dual_recovery_safety_iterations=0)
    with pytest.raises(ValueError, match="tokens"):
        replace(profile, recovery_joint_tokens=())


def test_unknown_fpa_profile_fails_explicitly() -> None:
    with pytest.raises(KeyError, match="Unknown FPA robot profile"):
        get_fpa_profile("kapex")
