"""Faithful Unitree H1 constants from the verified research DMR path."""

from __future__ import annotations

from math import radians
from types import MappingProxyType

from core_retarget.robots.profiles.schema import DmrProfile, IkSolverProfile

H1_JOI_BODY_NAMES = MappingProxyType(
    {
        "base": "pelvis",
        "lp": "left_hip_roll_link",
        "lk": "left_knee_link",
        "la": "left_ankle_link",
        "lf": "left_ankle_link",
        "lt": "left_toe_link",
        "lsole": "left_sole_link",
        "rp": "right_hip_roll_link",
        "rk": "right_knee_link",
        "ra": "right_ankle_link",
        "rf": "right_ankle_link",
        "rt": "right_toe_link",
        "rsole": "right_sole_link",
        "spine": "torso_link",
        "torso": "torso_link",
        "ls": "left_shoulder_roll_link",
        "le": "left_elbow_link_ball_hand",
        "lw": "left_hand_link",
        "lh": "left_hand_link",
        "lh_tip": "left_hand_tip_link",
        "rs": "right_shoulder_roll_link",
        "re": "right_elbow_link_ball_hand",
        "rw": "right_hand_link",
        "rh": "right_hand_link",
        "rh_tip": "right_hand_tip_link",
    }
)

H1_DMR_PROFILE = DmrProfile(
    robot_id="h1",
    qpos_dim=27,
    joi_bodies=H1_JOI_BODY_NAMES,
    wrist_joint_tokens=("wrist",),
    ankle_joint_tokens=("ankle",),
    toe_joint_tokens=("toe",),
    optimize_toe_dmr=True,
    link_length_base_reference="legacy_midhip",
    pelvis_orientation_reference_mode="source_absolute",
    pelvis_orientation_solve_stage="legacy_post",
    pelvis_orientation_weight=0.03,
    pelvis_orientation_axis_length=0.15,
    pelvis_orientation_smooth_time=0.05,
    ankle_orientation_mode="outsole_normal",
    ankle_orientation_stage="post",
    ankle_orientation_axes=(2,),
    ankle_orientation_axis_length=0.15,
    ankle_orientation_smooth_time=0.10,
    left_ankle_orientation_joi_key="lsole",
    right_ankle_orientation_joi_key="rsole",
    # Inherited from the K1 research profile.  The outsole-normal branch
    # calibrates from the neutral sole frames instead, so these constants are
    # retained for profile fidelity but are intentionally not consumed.
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
    left_hand_local_offset=None,
    right_hand_local_offset=None,
    left_hand_anchor_local=(0.0, 0.0, 0.0),
    right_hand_anchor_local=(0.0, 0.0, 0.0),
    left_hand_axis_signs=(1.0, 1.0, 1.0),
    right_hand_axis_signs=(1.0, 1.0, 1.0),
    hand_orientation_axis_length=0.10,
    hand_orientation_enabled=False,
    hand_orientation_reference_mode="first_realized",
    initial_warmup_passes=4,
    body_solver=IkSolverProfile(
        max_iterations=100,
        revolute_step=0.5,
        revolute_update_limit=radians(5.0),
        damping=1e-2,
        joint_limit_probe=radians(3.0),
    ),
    ankle_solver=IkSolverProfile(
        max_iterations=50,
        revolute_step=0.5,
        revolute_update_limit=radians(2.0),
        damping=1e-4,
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

__all__ = ["H1_DMR_PROFILE", "H1_JOI_BODY_NAMES"]
