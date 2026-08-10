"""LimX Oli DMR profile."""

from dataclasses import replace

from core_retarget.robots.joi.body import get_body_joi_mapping
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE

OLI_JOI_BODY_NAMES = get_body_joi_mapping("oli")

OLI_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="oli",
    qpos_dim=68,
    joi_bodies=OLI_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["OLI_DMR_PROFILE", "OLI_JOI_BODY_NAMES"]
