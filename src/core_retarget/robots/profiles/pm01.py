"""ENGINEAI PM01 DMR profile."""

from dataclasses import replace

from core_retarget.robots.joi.body import get_body_joi_mapping
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE

PM01_JOI_BODY_NAMES = get_body_joi_mapping("pm01")

PM01_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="pm01",
    qpos_dim=31,
    joi_bodies=PM01_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    wrist_joint_tokens=("ELBOW_YAW",),
    waist_joint_tokens=("J12_WAIST_YAW",),
    exclude_waist_from_primary_dmr=False,
    torso_orientation_weight=0.0,
    torso_orientation_stage="none",
    torso_orientation_axes=(),
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
    hand_orientation_enabled=False,
)

__all__ = ["PM01_DMR_PROFILE", "PM01_JOI_BODY_NAMES"]
