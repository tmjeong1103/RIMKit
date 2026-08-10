"""Fourier Intelligence N1 DMR profile."""

from dataclasses import replace

from core_retarget.robots.joi.body import get_body_joi_mapping
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE

N1_JOI_BODY_NAMES = get_body_joi_mapping("n1")

N1_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="n1",
    qpos_dim=30,
    joi_bodies=N1_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    waist_joint_tokens=("waist_yaw_joint",),
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["N1_DMR_PROFILE", "N1_JOI_BODY_NAMES"]
