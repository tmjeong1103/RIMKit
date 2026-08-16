"""Apptronik Apollo DMR profile."""

from dataclasses import replace

from rimkit.robots.joi.body import get_body_joi_mapping
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE

APOLLO_JOI_BODY_NAMES = get_body_joi_mapping("apollo")

APOLLO_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="apollo",
    qpos_dim=39,
    joi_bodies=APOLLO_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    waist_joint_tokens=("torso_roll", "torso_pitch"),
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["APOLLO_DMR_PROFILE", "APOLLO_JOI_BODY_NAMES"]
