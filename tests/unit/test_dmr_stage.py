from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import core_retarget.stages.dmr as dmr_stage
from core_retarget.exceptions import ConfigurationError, MotionValidationError
from core_retarget.motion import extract_soma_joi, load_soma_motion
from core_retarget.mujoco.model import MujocoModel
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE
from core_retarget.robots.profiles.h1 import H1_DMR_PROFILE
from core_retarget.robots.profiles.k1 import K1_DMR_PROFILE
from core_retarget.robots.profiles.r1 import R1_DMR_PROFILE

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
