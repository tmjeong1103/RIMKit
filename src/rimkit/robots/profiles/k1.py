"""ROBOTIS K1 DMR profile."""

from __future__ import annotations

from math import radians
from types import MappingProxyType

from rimkit.robots.profiles.schema import DmrProfile, IkSolverProfile

K1_JOI_BODY_NAMES = MappingProxyType(
    {
        "base": "pelvis",
        "lp": "left_hip_roll_link",
        "lk": "left_knee_link",
        "la": "left_ankle_roll_link",
        "lf": "left_ankle_roll_link",
        "lt": "left_toe_link",
        "rp": "right_hip_roll_link",
        "rk": "right_knee_link",
        "ra": "right_ankle_roll_link",
        "rf": "right_ankle_roll_link",
        "rt": "right_toe_link",
        "spine": "torso_link",
        "ls": "left_shoulder_roll_link",
        "le": "left_elbow_link",
        "lw": "left_wrist",
        "rs": "right_shoulder_roll_link",
        "re": "right_elbow_link",
        "rw": "right_wrist",
        "lh": "left_wrist",
        "rh": "right_wrist",
    }
)

K1_DMR_PROFILE = DmrProfile(
    robot_id="k1",
    qpos_dim=30,
    joi_bodies=K1_JOI_BODY_NAMES,
    wrist_joint_tokens=("wrist",),
    link_length_base_reference="legacy_midhip",
    pelvis_orientation_reference_mode="source_absolute",
    pelvis_orientation_solve_stage="legacy_post",
    pelvis_orientation_weight=0.03,
    pelvis_orientation_axis_length=0.15,
    pelvis_orientation_smooth_time=0.05,
    ankle_orientation_mode="source_body",
    ankle_orientation_stage="primary",
    ankle_orientation_axes=(1, 2),
    ankle_orientation_axis_length=0.10,
    left_ankle_local_offset=(
        (-0.12, 0.99, -0.02),
        (0.02, 0.02, 1.0),
        (0.99, 0.12, -0.02),
    ),
    right_ankle_local_offset=(
        (0.18, 0.98, 0.02),
        (0.05, -0.03, 1.0),
        (0.98, -0.18, -0.05),
    ),
    left_hand_local_offset=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    right_hand_local_offset=(
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    left_hand_anchor_local=(0.03, 0.0, 0.0),
    right_hand_anchor_local=(0.03, 0.0, 0.0),
    left_hand_axis_signs=(1.0, -1.0, 1.0),
    right_hand_axis_signs=(1.0, 1.0, 1.0),
    hand_orientation_axis_length=0.10,
    initial_warmup_passes=4,
    body_solver=IkSolverProfile(
        max_iterations=100,
        revolute_step=0.5,
        revolute_update_limit=radians(5.0),
        damping=1e-2,
        joint_limit_probe=radians(3.0),
    ),
    hand_solver=IkSolverProfile(
        max_iterations=10,
        revolute_step=0.5,
        revolute_update_limit=radians(10.0),
        damping=1e-4,
        joint_limit_probe=radians(3.0),
    ),
)

__all__ = ["K1_DMR_PROFILE", "K1_JOI_BODY_NAMES"]
