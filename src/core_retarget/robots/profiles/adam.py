"""PNDbotics ADAM Lite DMR profile."""

from dataclasses import replace

from core_retarget.robots.joi.body import get_body_joi_mapping
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE

ADAM_JOI_BODY_NAMES = get_body_joi_mapping("adam")

ADAM_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="adam",
    qpos_dim=32,
    joi_bodies=ADAM_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    waist_joint_tokens=("waistRoll", "waistPitch"),
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["ADAM_DMR_PROFILE", "ADAM_JOI_BODY_NAMES"]
