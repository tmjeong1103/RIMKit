"""Booster Robotics T1 DMR profile."""

from dataclasses import replace

from rimkit.robots.joi.body import get_body_joi_mapping
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE

T1_JOI_BODY_NAMES = get_body_joi_mapping("t1")

T1_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="t1",
    qpos_dim=30,
    joi_bodies=T1_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    wrist_joint_tokens=("Elbow_Yaw",),
    waist_joint_tokens=("Waist",),
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["T1_DMR_PROFILE", "T1_JOI_BODY_NAMES"]
