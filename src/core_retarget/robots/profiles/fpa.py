"""Immutable robot profiles for contact-aware foot pose adjustment.

The supported robots share one algorithm.  Differences are expressed as
data here, mirroring the ``RetargetRobotProfile`` values in the research
notebooks; the FPA implementation never branches on a robot identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, radians
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, slots=True)
class FpaProfile:
    """Numerical settings consumed by the FPA stages."""

    robot_id: str

    sole_target_mode: str = "shared_quantile"
    sole_clearance_quantile: float = 0.95
    sole_clearance_min: float = 0.002
    sole_clearance_max: float = 0.060
    sole_clearance_smooth_time: float = 0.0
    sole_ground_clearance: float = 0.001

    position_smooth_lambda: float = 1e-7
    position_remain_weight: float = 1e5
    yaw_smooth_lambda: float = 1e-8
    yaw_remain_weight: float = 1e4

    temporal_seed_blend: float = 0.0
    temporal_nullspace_gain: float = 0.0
    contact_weight_ramp_time: float = 0.10
    touchdown_preblend_time: float = 0.10
    touchdown_max_target_delta: float = 0.020
    toe_velocity_blend_time: float = 0.07
    toe_velocity_max_target_delta: float = 0.010

    joint_correction_median_window: int = 1
    joint_correction_smooth_time: float = 0.0
    joint_correction_max_delta: float = 0.0
    swing_outlier_threshold: float = inf
    swing_outlier_max_adjustment: float = 0.0
    swing_outlier_contact_threshold: float = 0.5

    ground_geometry_clearance: float = 0.001
    ground_geometry_max_correction: float = 0.03
    ground_geometry_ramp_speed: float = 0.03

    post_ground_recovery_passes: int = 0
    post_ground_recovery_smooth_time: float = 0.0
    post_ground_micro_lift_max: float = 0.0
    post_ground_micro_lift_speed: float = 0.0
    post_ground_micro_lift_include_swing_feet: bool = False
    post_ground_dual_support_lower_max: float = 0.0
    post_ground_dual_support_lower_speed: float = 0.10
    post_ground_root_lower_support_mode: str = "double"
    post_ground_dual_recovery_passes: int = 0
    post_ground_dual_recovery_joint_delta: float = radians(1.0)
    post_ground_dual_recovery_height_deadband: float = 0.0005
    post_ground_dual_recovery_min_root_lower: float = 0.0
    post_ground_dual_recovery_safety_iterations: int = 10

    recovery_joint_tokens: tuple[str, ...] = ("hip", "knee", "ankle")
    left_joint_tokens: tuple[str, ...] = ("left_",)
    right_joint_tokens: tuple[str, ...] = ("right_",)
    excluded_joint_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if self.sole_target_mode != "shared_quantile":
            raise ValueError("The supported FPA profiles use shared_quantile sole targets")
        if not 0.0 <= self.sole_clearance_quantile <= 1.0:
            raise ValueError("sole_clearance_quantile must lie in [0, 1]")
        if self.sole_clearance_min < 0.0:
            raise ValueError("sole_clearance_min must be non-negative")
        if self.sole_clearance_max < self.sole_clearance_min:
            raise ValueError("sole_clearance_max must not be below the minimum")
        if self.ground_geometry_ramp_speed <= 0.0:
            raise ValueError("ground_geometry_ramp_speed must be positive")
        if self.post_ground_root_lower_support_mode not in {"double", "any"}:
            raise ValueError("post_ground_root_lower_support_mode must be double or any")
        finite_nonnegative = (
            "sole_clearance_min",
            "sole_clearance_max",
            "sole_clearance_smooth_time",
            "sole_ground_clearance",
            "position_smooth_lambda",
            "position_remain_weight",
            "yaw_smooth_lambda",
            "yaw_remain_weight",
            "temporal_seed_blend",
            "temporal_nullspace_gain",
            "contact_weight_ramp_time",
            "touchdown_preblend_time",
            "touchdown_max_target_delta",
            "toe_velocity_blend_time",
            "toe_velocity_max_target_delta",
            "joint_correction_smooth_time",
            "joint_correction_max_delta",
            "swing_outlier_max_adjustment",
            "ground_geometry_clearance",
            "ground_geometry_max_correction",
            "post_ground_recovery_smooth_time",
            "post_ground_micro_lift_max",
            "post_ground_micro_lift_speed",
            "post_ground_dual_support_lower_max",
            "post_ground_dual_recovery_joint_delta",
            "post_ground_dual_recovery_height_deadband",
            "post_ground_dual_recovery_min_root_lower",
        )
        for field_name in finite_nonnegative:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not np.isfinite(self.swing_outlier_threshold) and self.swing_outlier_threshold != inf:
            raise ValueError("swing_outlier_threshold must be non-negative or infinity")
        if self.swing_outlier_threshold < 0.0:
            raise ValueError("swing_outlier_threshold must be non-negative or infinity")
        if not 0.0 <= self.swing_outlier_contact_threshold <= 1.0:
            raise ValueError("swing_outlier_contact_threshold must lie in [0, 1]")
        if self.joint_correction_median_window < 1:
            raise ValueError("joint_correction_median_window must be positive")
        if self.post_ground_recovery_passes < 0:
            raise ValueError("post_ground_recovery_passes must be non-negative")
        if self.post_ground_dual_recovery_passes < 0:
            raise ValueError("post_ground_dual_recovery_passes must be non-negative")
        if self.post_ground_dual_recovery_safety_iterations < 1:
            raise ValueError("post_ground_dual_recovery_safety_iterations must be positive")
        for field_name in (
            "recovery_joint_tokens",
            "left_joint_tokens",
            "right_joint_tokens",
        ):
            values = tuple(str(value) for value in getattr(self, field_name))
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty tokens")
            object.__setattr__(self, field_name, values)
        excluded = tuple(str(value) for value in self.excluded_joint_tokens)
        if any(not value.strip() for value in excluded):
            raise ValueError("excluded_joint_tokens must not contain empty tokens")
        object.__setattr__(self, "excluded_joint_tokens", excluded)


_K1 = FpaProfile(robot_id="k1")
_G1 = FpaProfile(
    robot_id="g1",
    contact_weight_ramp_time=0.14,
    touchdown_preblend_time=0.14,
    touchdown_max_target_delta=0.030,
    toe_velocity_blend_time=0.12,
    joint_correction_median_window=3,
    joint_correction_smooth_time=0.06,
    joint_correction_max_delta=radians(0.25),
    swing_outlier_threshold=radians(10.0),
    swing_outlier_max_adjustment=radians(24.0),
    post_ground_dual_support_lower_max=0.012,
    post_ground_dual_support_lower_speed=0.10,
    post_ground_dual_recovery_passes=2,
    post_ground_dual_recovery_joint_delta=radians(1.0),
    post_ground_dual_recovery_height_deadband=0.0005,
    post_ground_dual_recovery_min_root_lower=0.0,
    post_ground_dual_recovery_safety_iterations=10,
)

FPA_PROFILES = MappingProxyType(
    {
        "k1": _K1,
        "h1": replace(_K1, robot_id="h1"),
        "g1": _G1,
        "h2": replace(_G1, robot_id="h2"),
        "r1": replace(_G1, robot_id="r1"),
        "apollo": replace(
            _G1,
            robot_id="apollo",
            left_joint_tokens=("l_",),
            right_joint_tokens=("r_",),
        ),
        "oli": replace(_G1, robot_id="oli"),
        "n1": replace(_G1, robot_id="n1"),
        "adam": replace(
            _G1,
            robot_id="adam",
            left_joint_tokens=("_Left",),
            right_joint_tokens=("_Right",),
        ),
        "t1": replace(
            _G1,
            robot_id="t1",
            left_joint_tokens=("Left_",),
            right_joint_tokens=("Right_",),
            excluded_joint_tokens=("Waist",),
            post_ground_micro_lift_max=0.040,
            post_ground_micro_lift_speed=0.20,
            post_ground_micro_lift_include_swing_feet=False,
        ),
        "pm01": replace(
            _G1,
            robot_id="pm01",
            left_joint_tokens=("_L",),
            right_joint_tokens=("_R",),
            excluded_joint_tokens=("J12_WAIST_YAW",),
            post_ground_micro_lift_max=0.010,
            post_ground_micro_lift_speed=0.20,
            post_ground_micro_lift_include_swing_feet=False,
        ),
    }
)


def get_fpa_profile(robot_id: str) -> FpaProfile:
    """Return the immutable FPA profile for one supported robot."""

    key = str(robot_id).strip().lower()
    try:
        return FPA_PROFILES[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown FPA robot profile {robot_id!r}; available={tuple(FPA_PROFILES)}"
        ) from exc


__all__ = ["FPA_PROFILES", "FpaProfile", "get_fpa_profile"]
