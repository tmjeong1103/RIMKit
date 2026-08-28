"""AgiBot X2-Ultra DMR profile."""

from dataclasses import replace

from rimkit.robots.joi.body import get_body_joi_mapping
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE

X2_JOI_BODY_NAMES = get_body_joi_mapping("x2")

X2_DMR_PROFILE = replace(
    G1_DMR_PROFILE,
    robot_id="x2",
    qpos_dim=38,
    joi_bodies=X2_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    torso_orientation_joi_key="torso",
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
)

__all__ = ["X2_DMR_PROFILE", "X2_JOI_BODY_NAMES"]
