from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import rimkit.stages.dmr as dmr_stage
from rimkit.exceptions import ConfigurationError, MotionValidationError
from rimkit.motion import extract_soma_joi, load_soma_motion
from rimkit.mujoco.model import MujocoModel
from rimkit.mujoco.robot_kinematics import derive_neutral_geometry
from rimkit.robots.profiles.a3 import A3_DMR_PROFILE
from rimkit.robots.profiles.asimov1 import ASIMOV1_DMR_PROFILE
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE
from rimkit.robots.profiles.h1 import H1_DMR_PROFILE
from rimkit.robots.profiles.k1 import K1_DMR_PROFILE
from rimkit.robots.profiles.r1 import R1_DMR_PROFILE
from rimkit.robots.profiles.x2 import X2_DMR_PROFILE

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "motions" / "kimodo" / "soma_rp_v11" / "stand_walk_run_stop.npz"


def _model_with_joints(*names: str) -> MujocoModel:
    return cast(MujocoModel, SimpleNamespace(rev_pri_joint_names=names))


def test_joint_groups_honor_optimize_toe_dmr() -> None:
    model = _model_with_joints(
        "left_hip_joint",
        "left_toe_joint",
        "left_wrist_joint",
        "waist_yaw_joint",
    )

    optimized = dmr_stage._joint_groups(model, K1_DMR_PROFILE)
    fixed = dmr_stage._joint_groups(
        model,
        replace(K1_DMR_PROFILE, optimize_toe_dmr=False),
    )

    assert optimized.toe == ("left_toe_joint",)
    assert "left_toe_joint" in optimized.body
    assert "left_toe_joint" not in fixed.body
    assert fixed.body == ("left_hip_joint", "waist_yaw_joint")


@pytest.mark.parametrize(
    ("groups", "message"),
    (
        (
            dmr_stage._JointGroups(body=(), wrist=("wrist",), waist=(), ankle=("ankle",), toe=()),
            "torso post solver.*waist_joint_tokens matched no model joints",
        ),
        (
            dmr_stage._JointGroups(body=(), wrist=("wrist",), waist=("waist",), ankle=(), toe=()),
            "ankle post solver.*ankle_joint_tokens matched no model joints",
        ),
        (
            dmr_stage._JointGroups(body=(), wrist=(), waist=("waist",), ankle=("ankle",), toe=()),
            "hand orientation.*wrist_joint_tokens matched no model joints",
        ),
    ),
)
def test_active_solver_groups_must_not_be_empty(
    groups: dmr_stage._JointGroups,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        dmr_stage._validate_active_joint_groups(G1_DMR_PROFILE, groups)


def test_disabled_hand_orientation_accepts_h1_without_wrist_joints() -> None:
    groups = dmr_stage._JointGroups(
        body=("not_use_joint",),
        wrist=(),
        waist=(),
        ankle=("left_ankle_joint", "right_ankle_joint"),
        toe=(),
    )

    dmr_stage._validate_active_joint_groups(H1_DMR_PROFILE, groups)

    with pytest.raises(ConfigurationError, match="hand orientation"):
        dmr_stage._validate_active_joint_groups(
            replace(H1_DMR_PROFILE, hand_orientation_enabled=True),
            groups,
        )


def test_robot_neutral_delta_is_conjugated_into_pelvis_body_frame() -> None:
    body_from_anatomical = dmr_stage.Rotation.from_euler("xyz", (0.21, -0.13, 0.37)).as_matrix()
    source_reference = np.array([0.0, 0.0, 1.0])
    source_current = np.array([0.24, -0.18, 0.954777])
    source_current /= np.linalg.norm(source_current)
    robot_local = np.array([0.08, -0.04, 0.52])

    anatomical_delta = dmr_stage._minimal_rotation_between(
        source_reference,
        source_current,
    )
    expected = body_from_anatomical.T @ anatomical_delta @ body_from_anatomical @ robot_local
    actual = dmr_stage._robot_neutral_delta_vector(
        robot_local,
        source_reference,
        source_current,
        body_from_anatomical,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_run_dmr_rejects_source_joi_timestamp_mismatch() -> None:
    motion = load_soma_motion(EXAMPLE)
    source_joi = extract_soma_joi(motion)
    mismatched_joi = replace(source_joi, seconds=source_joi.seconds + 1e-3)

    with pytest.raises(MotionValidationError, match="JOI trajectory timestamps"):
        dmr_stage.run_dmr(motion, robot_id="k1", source_joi=mismatched_joi)


@pytest.mark.mujoco
@pytest.mark.parametrize("robot_id", ("g1", "h2", "r1"))
def test_contact_aware_dmr_requires_at_least_two_frames(robot_id: str) -> None:
    motion = load_soma_motion(EXAMPLE)
    one_frame = replace(
        motion,
        summary=replace(
            motion.summary,
            frame_count=1,
            duration_seconds=1.0 / motion.fps,
        ),
        seconds=motion.seconds[:1],
        posed_joints=motion.posed_joints[:1],
        global_rot_mats=motion.global_rot_mats[:1],
        foot_contacts=None if motion.foot_contacts is None else motion.foot_contacts[:1],
    )

    with pytest.raises(MotionValidationError, match="requires at least two frames"):
        dmr_stage.run_dmr(one_frame, robot_id=robot_id)


@pytest.mark.mujoco
def test_r1_semantic_base_anchor_is_the_neutral_auxiliary_hip_midpoint() -> None:
    model = MujocoModel.from_robot("r1")
    anchors = dmr_stage._resolve_semantic_joi_anchors(model, R1_DMR_PROFILE)

    left_hip = model.get_body_transform(R1_DMR_PROFILE.joi_bodies["lp"])[:3, 3]
    right_hip = model.get_body_transform(R1_DMR_PROFILE.joi_bodies["rp"])[:3, 3]
    expected_position = 0.5 * (left_hip + right_hip)
    actual_position = dmr_stage._semantic_joi_position(
        model,
        R1_DMR_PROFILE,
        "base",
        anchors,
    )
    actual_transform = dmr_stage._semantic_joi_transform(
        model,
        R1_DMR_PROFILE,
        "base",
        anchors,
    )

    assert tuple(anchors) == ("base",)
    assert anchors["base"] == pytest.approx(
        np.array([0.0325, 0.0, -0.157287248407]),
        abs=1e-12,
    )
    np.testing.assert_allclose(actual_position, expected_position, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(actual_transform[:3, 3], expected_position, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(
        actual_transform[:3, :3],
        model.get_body_transform(R1_DMR_PROFILE.joi_bodies["base"])[:3, :3],
    )


@pytest.mark.mujoco
def test_robot_bind_trunk_uses_semantic_base_anchor_for_a3() -> None:
    model = MujocoModel.from_robot("a3")
    model.reset()
    geometry = derive_neutral_geometry(
        model,
        A3_DMR_PROFILE.joi_bodies,
        link_length_base_reference=A3_DMR_PROFILE.link_length_base_reference,
    )
    anchors = dmr_stage._resolve_semantic_joi_anchors(model, A3_DMR_PROFILE)
    base_position = dmr_stage._semantic_joi_position(
        model,
        A3_DMR_PROFILE,
        "base",
        anchors,
    )
    base_rotation = geometry.body_transforms["base"][:3, :3]
    robot_spine_local = base_rotation.T @ (geometry.body_transforms["spine"][:3, 3] - base_position)
    robot_neck_local = base_rotation.T @ (geometry.body_transforms["neck"][:3, 3] - base_position)

    source = dict(geometry.body_transforms)
    source_base = np.array(source["base"], copy=True)
    source_base[:3, 3] = base_position
    source["base"] = source_base
    source["rtoe"] = source["rt"]
    source["ltoe"] = source["lt"]
    targets = dict(
        dmr_stage._body_targets(
            source,
            geometry,
            profile=A3_DMR_PROFILE,
            effective_base_rotation=base_rotation,
            trunk_blend=1.0,
            source_base_rotation=base_rotation,
            source_spine_reference_local=np.array([0.0, 0.0, 1.0]),
            source_neck_reference_local=np.array([0.0, 0.0, 1.0]),
            robot_spine_local=robot_spine_local,
            robot_neck_local=robot_neck_local,
            robot_body_from_anatomical=np.eye(3),
        )
    )

    np.testing.assert_allclose(
        targets["spine"],
        geometry.body_transforms["spine"][:3, 3],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.mujoco
def test_asimov1_handless_profile_runs_wrist_position_only() -> None:
    motion = load_soma_motion(EXAMPLE)
    frame_count = 2
    short_motion = replace(
        motion,
        summary=replace(
            motion.summary,
            frame_count=frame_count,
            duration_seconds=frame_count / motion.fps,
        ),
        seconds=motion.seconds[:frame_count],
        posed_joints=motion.posed_joints[:frame_count],
        global_rot_mats=motion.global_rot_mats[:frame_count],
        foot_contacts=(
            None if motion.foot_contacts is None else motion.foot_contacts[:frame_count]
        ),
    )

    result = dmr_stage.run_dmr(short_motion, robot_id="asimov1", backend="python")

    assert result.qpos.shape == (frame_count, ASIMOV1_DMR_PROFILE.qpos_dim)
    assert np.isfinite(result.qpos).all()
    assert "lh" not in ASIMOV1_DMR_PROFILE.joi_bodies
    assert "rh" not in ASIMOV1_DMR_PROFILE.joi_bodies
    assert not ASIMOV1_DMR_PROFILE.hand_orientation_enabled


@pytest.mark.mujoco
def test_x2_profile_resolves_all_six_wrist_orientation_joints() -> None:
    groups = dmr_stage._joint_groups(MujocoModel.from_robot("x2"), X2_DMR_PROFILE)

    assert groups.wrist == (
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    )
    assert X2_DMR_PROFILE.hand_orientation_enabled
    assert X2_DMR_PROFILE.joi_bodies["lh"] == "left_hand_link"
    assert X2_DMR_PROFILE.joi_bodies["rh"] == "right_hand_link"


@pytest.mark.mujoco
def test_arm_recovery_reseeds_only_primary_joints_on_one_wrist_chain() -> None:
    model = MujocoModel.from_robot("g1")
    groups = dmr_stage._joint_groups(model, G1_DMR_PROFILE)

    arm_joints = dmr_stage._arm_joint_names(model, G1_DMR_PROFILE, groups.body)

    assert set(arm_joints) == {
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
    }
    assert set(arm_joints) <= set(groups.body)
    assert not set(arm_joints) & set(groups.wrist)
