"""Immutable robot-specific settings consumed by the DMR stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

Matrix3 = tuple[tuple[float, ...], ...]
Vector3 = tuple[float, float, float]


def _matrix3(value: Sequence[Sequence[float]], *, field_name: str) -> Matrix3:
    matrix = tuple(tuple(float(component) for component in row) for row in value)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"{field_name} must be a 3-by-3 matrix.")
    return matrix


def _vector3(value: Sequence[float], *, field_name: str) -> Vector3:
    vector = tuple(float(component) for component in value)
    if len(vector) != 3:
        raise ValueError(f"{field_name} must contain exactly three values.")
    return vector[0], vector[1], vector[2]


@dataclass(frozen=True, slots=True)
class IkSolverProfile:
    """Numerical settings for one inverse-kinematics solve."""

    max_iterations: int
    revolute_step: float
    revolute_update_limit: float
    damping: float
    joint_limit_probe: float

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        nonnegative_fields = (
            "revolute_step",
            "revolute_update_limit",
            "damping",
            "joint_limit_probe",
        )
        for field_name in nonnegative_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class InitialCollisionProfile:
    """Immutable settings for the first collision-refinement stage."""

    robot_id: str
    qpos_dim: int
    solver: IkSolverProfile

    initial_margin: float = 0.02
    correction_gain: float = 0.5
    ticks_per_pass: int = 24
    correction_length_cap: float = 0.03
    outer_passes: int = 4
    margin_scale: float = 1.4
    margin_cap: float = 0.03
    query_limit: int = 32
    target_limit: int = 16
    ancestor_skip_depth: int = 2
    root_body_name: str = "pelvis"
    movable_joint_tokens: tuple[str, ...] = (
        "shoulder",
        "elbow",
        "wrist",
        "arm",
    )
    movable_body_tokens: tuple[str, ...] = (
        "shoulder",
        "elbow",
        "wrist",
        "hand",
        "palm",
    )
    movable_body_prefixes: tuple[str, ...] = ("left_", "right_")
    preserve_orientation: bool = True
    orientation_weight: float = 0.03
    orientation_axis_length: float = 0.08
    smooth_each_pass: bool = True
    final_pass_without_smoothing: bool = True
    final_pass_margin: float = 0.002
    smooth_jerk_weight: float = 1e-5
    smooth_tracking_norm: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "movable_joint_tokens",
            "movable_body_tokens",
            "movable_body_prefixes",
        ):
            values = tuple(str(value) for value in getattr(self, field_name))
            object.__setattr__(self, field_name, values)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain empty values.")

        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty.")
        if self.qpos_dim <= 0:
            raise ValueError("qpos_dim must be positive.")
        if self.ticks_per_pass <= 0 or self.outer_passes <= 0:
            raise ValueError("collision iteration counts must be positive.")
        if self.query_limit <= 0 or self.target_limit <= 0:
            raise ValueError("collision top-k limits must be positive.")
        if self.target_limit > 2 * self.query_limit:
            raise ValueError("target_limit cannot exceed two targets per queried pair.")
        if self.ancestor_skip_depth < 0:
            raise ValueError("ancestor_skip_depth must be non-negative.")
        if not self.root_body_name.strip():
            raise ValueError("root_body_name must not be empty.")
        for field_name in (
            "initial_margin",
            "correction_gain",
            "correction_length_cap",
            "margin_scale",
            "margin_cap",
            "orientation_weight",
            "orientation_axis_length",
            "final_pass_margin",
            "smooth_jerk_weight",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.margin_scale <= 0.0:
            raise ValueError("margin_scale must be positive.")
        if self.smooth_tracking_norm not in (1, 2):
            raise ValueError("smooth_tracking_norm must be 1 or 2.")


@dataclass(frozen=True, slots=True)
class DmrProfile:
    """Robot metadata and tuning required by the Phase-2 DMR implementation.

    Every collection is copied into an immutable representation during
    construction.  A caller therefore cannot mutate a registered profile via
    either the original input objects or a value returned by the registry.
    """

    robot_id: str
    qpos_dim: int
    joi_bodies: Mapping[str, str]
    wrist_joint_tokens: tuple[str, ...]
    link_length_base_reference: str

    pelvis_orientation_reference_mode: str
    pelvis_orientation_solve_stage: str
    pelvis_orientation_weight: float
    pelvis_orientation_axis_length: float
    pelvis_orientation_smooth_time: float

    ankle_orientation_mode: str
    ankle_orientation_stage: str
    ankle_orientation_axes: tuple[int, ...]
    ankle_orientation_axis_length: float
    left_ankle_local_offset: Matrix3
    right_ankle_local_offset: Matrix3

    left_hand_local_offset: Matrix3 | None
    right_hand_local_offset: Matrix3 | None
    left_hand_anchor_local: Vector3
    right_hand_anchor_local: Vector3
    left_hand_axis_signs: Vector3
    right_hand_axis_signs: Vector3
    hand_orientation_axis_length: float

    initial_warmup_passes: int
    body_solver: IkSolverProfile
    hand_solver: IkSolverProfile

    # Some robots expose only fixed hand frames and no actuated wrist chain.
    # Those profiles retain the position targets but skip the orientation solve.
    hand_orientation_enabled: bool = True

    # Optional semantic points rigidly attached to mapped MuJoCo bodies.  Each
    # entry maps a JOI key to other JOI keys whose neutral-position mean
    # defines the point.  This preserves model-derived landmarks without
    # hard-coding robot-specific offsets in the DMR engine.
    joi_anchor_reference_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    # Joint groups are resolved from MuJoCo joint-name tokens.  G1 and H2 keep
    # their yaw waist joint in the primary solve while assigning roll/pitch to
    # the fixed-base torso pass.
    waist_joint_tokens: tuple[str, ...] = ("waist",)
    ankle_joint_tokens: tuple[str, ...] = ("ankle",)
    toe_joint_tokens: tuple[str, ...] = ("toe",)
    exclude_waist_from_primary_dmr: bool = False
    optimize_toe_dmr: bool = True

    # Primary pelvis tracking and contact-aware stabilization.  The defaults
    # retain the verified K1 path; G1, H2, and R1 enable it explicitly.
    pelvis_primary_orientation_weight: float = 0.0
    pelvis_primary_dynamic_orientation_weight: float | None = None
    pelvis_stabilization_strength: float = 0.0
    pelvis_stabilization_orientation_weight: float = 0.0
    pelvis_stabilization_linear_speed_low: float = 0.08
    pelvis_stabilization_linear_speed_high: float = 0.25
    pelvis_stabilization_angular_speed_low: float = 0.15
    pelvis_stabilization_angular_speed_high: float = 0.80
    pelvis_stabilization_smooth_time: float = 0.10
    pelvis_support_transition_time: float = 0.0

    # Contact-gated central-trunk position reconstruction.
    trunk_position_mode: str = "source_world"
    trunk_position_gate: str = "stability"
    trunk_position_strength: float = 0.0

    # Optional articulated-torso task.  ``torso_solver`` is required only
    # when the stage is active.
    torso_orientation_weight: float = 0.0
    torso_orientation_stage: str = "none"
    torso_orientation_joi_key: str = "spine"
    torso_orientation_reference_mode: str = "absolute"
    torso_orientation_axes: tuple[int, ...] = ()
    torso_orientation_axis_length: float = 0.15
    torso_orientation_smooth_time: float = 0.05
    torso_solver: IkSolverProfile | None = None

    # Post ankle orientation uses its own fixed-base solver.  Source-body
    # K1 targets remain part of the primary solve and therefore need neither.
    ankle_orientation_smooth_time: float = 0.0
    left_ankle_orientation_joi_key: str | None = None
    right_ankle_orientation_joi_key: str | None = None
    ankle_solver: IkSolverProfile | None = None

    # ``None`` hand offsets are calibrated after the first realized position
    # solve.
    hand_orientation_reference_mode: str = "first_realized"

    # A previous solved pose can act as a weak null-space reference without
    # changing the task-space endpoint targets.
    dmr_initial_nullspace_gain: float = 0.0
    dmr_temporal_nullspace_gain: float = 0.0

    # Source-specific orientation filtering and semantic-sole stabilization.
    # The defaults preserve the frozen Kimodo path; GEM-X overlays opt into
    # sign-continuous quaternion filtering and contact flattening.
    orientation_smoothing_mode: str = "rotvec_legacy"
    ankle_contact_flatten_strength: float = 0.0
    ankle_contact_flatten_smooth_time: float = 0.0

    # Optional offline filtering of selected articulated joints after all
    # per-frame IK solves have completed.
    pelvis_stabilization_joint_smooth_tokens: tuple[str, ...] = (
        "hip",
        "knee",
        "ankle",
    )
    pelvis_stabilization_joint_median_window: int = 1
    pelvis_stabilization_joint_smooth_time: float = 0.0
    pelvis_stabilization_joint_smooth_max_delta: float = 0.0
    pelvis_stabilization_joint_smooth_gate: str = "stability"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joi_bodies",
            MappingProxyType(dict(self.joi_bodies)),
        )
        object.__setattr__(
            self,
            "joi_anchor_reference_keys",
            MappingProxyType(
                {
                    str(key): tuple(str(reference) for reference in references)
                    for key, references in self.joi_anchor_reference_keys.items()
                }
            ),
        )
        object.__setattr__(self, "wrist_joint_tokens", tuple(self.wrist_joint_tokens))
        for field_name in (
            "waist_joint_tokens",
            "ankle_joint_tokens",
            "toe_joint_tokens",
            "pelvis_stabilization_joint_smooth_tokens",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "ankle_orientation_axes",
            tuple(int(axis) for axis in self.ankle_orientation_axes),
        )
        object.__setattr__(
            self,
            "torso_orientation_axes",
            tuple(int(axis) for axis in self.torso_orientation_axes),
        )

        for field_name in (
            "left_ankle_local_offset",
            "right_ankle_local_offset",
            "left_hand_local_offset",
            "right_hand_local_offset",
        ):
            field_value = getattr(self, field_name)
            if field_value is None:
                if field_name.startswith(("left_ankle", "right_ankle")):
                    raise ValueError(f"{field_name} must not be None.")
                continue
            object.__setattr__(
                self,
                field_name,
                _matrix3(field_value, field_name=field_name),
            )
        for field_name in (
            "left_hand_anchor_local",
            "right_hand_anchor_local",
            "left_hand_axis_signs",
            "right_hand_axis_signs",
        ):
            object.__setattr__(
                self,
                field_name,
                _vector3(getattr(self, field_name), field_name=field_name),
            )

        if self.qpos_dim <= 0:
            raise ValueError("qpos_dim must be positive.")
        if self.initial_warmup_passes <= 0:
            raise ValueError("initial_warmup_passes must be positive.")
        if not self.joi_bodies:
            raise ValueError("joi_bodies must not be empty.")
        for key, references in self.joi_anchor_reference_keys.items():
            if key not in self.joi_bodies:
                raise ValueError(f"Semantic JOI anchor {key!r} is missing from joi_bodies.")
            if not references:
                raise ValueError(f"Semantic JOI anchor {key!r} must have reference keys.")
            missing_references = tuple(
                reference for reference in references if reference not in self.joi_bodies
            )
            if missing_references:
                raise ValueError(
                    f"Semantic JOI anchor {key!r} has unknown reference keys: "
                    + ", ".join(repr(reference) for reference in missing_references)
                    + "."
                )
        if any(axis not in (0, 1, 2) for axis in self.ankle_orientation_axes):
            raise ValueError("ankle_orientation_axes may only contain 0, 1, and 2.")
        if any(axis not in (0, 1, 2) for axis in self.torso_orientation_axes):
            raise ValueError("torso_orientation_axes may only contain 0, 1, and 2.")

        token_fields = (
            "wrist_joint_tokens",
            "waist_joint_tokens",
            "ankle_joint_tokens",
            "toe_joint_tokens",
            "pelvis_stabilization_joint_smooth_tokens",
        )
        for field_name in token_fields:
            values = getattr(self, field_name)
            if any(not str(token).strip() for token in values):
                raise ValueError(f"{field_name} must not contain empty tokens.")

        if self.link_length_base_reference not in {"body_origin", "legacy_midhip"}:
            raise ValueError("link_length_base_reference must be 'body_origin' or 'legacy_midhip'.")
        if self.pelvis_orientation_reference_mode not in {
            "source_absolute",
            "robot_neutral_delta",
        }:
            raise ValueError("Unsupported pelvis_orientation_reference_mode.")
        if self.pelvis_orientation_solve_stage not in {"legacy_post", "primary"}:
            raise ValueError("Unsupported pelvis_orientation_solve_stage.")
        if (
            self.pelvis_orientation_solve_stage == "primary"
            and self.pelvis_primary_orientation_weight <= 0.0
        ):
            raise ValueError("Primary pelvis orientation requires a positive weight.")

        nonnegative_fields = (
            "pelvis_orientation_weight",
            "pelvis_orientation_axis_length",
            "pelvis_orientation_smooth_time",
            "pelvis_primary_orientation_weight",
            "pelvis_stabilization_orientation_weight",
            "pelvis_stabilization_linear_speed_low",
            "pelvis_stabilization_angular_speed_low",
            "pelvis_stabilization_smooth_time",
            "pelvis_support_transition_time",
            "torso_orientation_weight",
            "torso_orientation_axis_length",
            "torso_orientation_smooth_time",
            "ankle_orientation_axis_length",
            "ankle_orientation_smooth_time",
            "hand_orientation_axis_length",
            "dmr_initial_nullspace_gain",
            "dmr_temporal_nullspace_gain",
            "ankle_contact_flatten_smooth_time",
            "pelvis_stabilization_joint_smooth_time",
            "pelvis_stabilization_joint_smooth_max_delta",
        )
        for field_name in nonnegative_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.orientation_smoothing_mode not in {
            "rotvec_legacy",
            "quaternion_continuous",
        }:
            raise ValueError("Unsupported orientation_smoothing_mode.")
        if not 0.0 <= self.ankle_contact_flatten_strength <= 1.0:
            raise ValueError("ankle_contact_flatten_strength must be in [0, 1].")
        if (
            self.pelvis_primary_dynamic_orientation_weight is not None
            and self.pelvis_primary_dynamic_orientation_weight < 0.0
        ):
            raise ValueError("pelvis_primary_dynamic_orientation_weight must be non-negative.")
        if not 0.0 <= self.pelvis_stabilization_strength <= 1.0:
            raise ValueError("pelvis_stabilization_strength must be in [0, 1].")
        if (
            self.pelvis_stabilization_linear_speed_high
            <= self.pelvis_stabilization_linear_speed_low
        ):
            raise ValueError("pelvis stabilization linear-speed high must exceed low.")
        if (
            self.pelvis_stabilization_angular_speed_high
            <= self.pelvis_stabilization_angular_speed_low
        ):
            raise ValueError("pelvis stabilization angular-speed high must exceed low.")

        if self.trunk_position_mode not in {
            "source_world",
            "robot_bind_local",
            "robot_neutral_delta",
        }:
            raise ValueError("Unsupported trunk_position_mode.")
        if self.trunk_position_gate not in {"stability", "always"}:
            raise ValueError("Unsupported trunk_position_gate.")
        if not 0.0 <= self.trunk_position_strength <= 1.0:
            raise ValueError("trunk_position_strength must be in [0, 1].")

        if self.torso_orientation_stage not in {"none", "post"}:
            raise ValueError("Unsupported torso_orientation_stage.")
        if self.torso_orientation_reference_mode not in {"absolute", "source_delta"}:
            raise ValueError("Unsupported torso_orientation_reference_mode.")
        if not str(self.torso_orientation_joi_key).strip():
            raise ValueError("torso_orientation_joi_key must not be empty.")
        if self.torso_orientation_stage == "post":
            if self.torso_orientation_weight <= 0.0:
                raise ValueError("Post torso orientation requires a positive weight.")
            if self.torso_solver is None:
                raise ValueError("Post torso orientation requires torso_solver.")
            if self.torso_orientation_joi_key not in self.joi_bodies:
                raise ValueError("torso_orientation_joi_key is missing from joi_bodies.")
        elif self.exclude_waist_from_primary_dmr:
            raise ValueError("Waist joints may be excluded only with a post torso stage.")

        if self.ankle_orientation_mode not in {
            "none",
            "source_body",
            "outsole_normal",
        }:
            raise ValueError("Unsupported ankle_orientation_mode.")
        if self.ankle_orientation_stage not in {"none", "primary", "post"}:
            raise ValueError("Unsupported ankle_orientation_stage.")
        if self.ankle_orientation_mode == "none" and self.ankle_orientation_stage != "none":
            raise ValueError("Ankle stage must be 'none' when orientation is disabled.")
        if self.ankle_orientation_stage == "post" and self.ankle_solver is None:
            raise ValueError("Post ankle orientation requires ankle_solver.")
        for field_name in (
            "left_ankle_orientation_joi_key",
            "right_ankle_orientation_joi_key",
        ):
            value = getattr(self, field_name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{field_name} must not be empty.")

        if self.hand_orientation_reference_mode != "first_realized":
            raise ValueError("Unsupported hand_orientation_reference_mode.")
        if self.pelvis_stabilization_joint_median_window <= 0:
            raise ValueError("pelvis_stabilization_joint_median_window must be positive.")
        if self.pelvis_stabilization_joint_smooth_gate not in {"stability", "always"}:
            raise ValueError("Unsupported pelvis_stabilization_joint_smooth_gate.")


__all__ = [
    "DmrProfile",
    "IkSolverProfile",
    "InitialCollisionProfile",
    "Matrix3",
    "Vector3",
]
