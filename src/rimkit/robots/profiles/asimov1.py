"""Asimov-1 DMR profile."""

from dataclasses import replace

from rimkit.robots.joi.body import get_body_joi_mapping
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE
from rimkit.robots.profiles.k1 import K1_DMR_PROFILE

ASIMOV1_JOI_BODY_NAMES = get_body_joi_mapping("asimov1")

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

ASIMOV1_DMR_PROFILE = replace(
    K1_DMR_PROFILE,
    robot_id="asimov1",
    qpos_dim=30,
    joi_bodies=ASIMOV1_JOI_BODY_NAMES,
    joi_anchor_reference_keys={"base": ("lp", "rp")},
    wrist_joint_tokens=("wrist_yaw",),
    waist_joint_tokens=("waist_yaw",),
    ankle_orientation_mode="outsole_normal",
    ankle_orientation_stage="post",
    ankle_orientation_axes=(2,),
    ankle_orientation_axis_length=0.15,
    ankle_orientation_smooth_time=0.10,
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
    left_ankle_local_offset=_IDENTITY,
    right_ankle_local_offset=_IDENTITY,
    ankle_solver=G1_DMR_PROFILE.ankle_solver,
    left_hand_local_offset=None,
    right_hand_local_offset=None,
    hand_orientation_enabled=False,
)

__all__ = ["ASIMOV1_DMR_PROFILE", "ASIMOV1_JOI_BODY_NAMES"]
