from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import rimkit.stages.dmr as dmr_stage
from rimkit.exceptions import ConfigurationError, MotionValidationError
from rimkit.motion import extract_soma_joi, load_soma_motion
from rimkit.motion.soma_joints import SOMA77_JOINT_NAMES
from rimkit.mujoco.model import MujocoModel
from rimkit.mujoco.robot_kinematics import derive_neutral_geometry
from rimkit.robots.profiles.a3 import A3_DMR_PROFILE
from rimkit.robots.profiles.asimov1 import ASIMOV1_DMR_PROFILE
from rimkit.robots.profiles.g1 import G1_DMR_PROFILE
from rimkit.robots.profiles.h1 import H1_DMR_PROFILE
from rimkit.robots.profiles.k1 import K1_DMR_PROFILE
from rimkit.robots.profiles.r1 import R1_DMR_PROFILE
from rimkit.robots.profiles.schema import DmrProfile
from rimkit.robots.profiles.t1 import T1_DMR_PROFILE
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
def test_arm_stage_solves_the_shoulder_and_elbow_joints() -> None:
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


def test_arm_spike_repair_removes_branch_switches_and_keeps_fast_motion() -> None:
    dt = 1.0 / 30.0
    seconds = np.arange(90) * dt
    # A fast 3 Hz arm swing, with a two-frame IK branch switch on top of it.
    smooth = 0.8 * np.sin(2.0 * np.pi * 3.0 * seconds)
    values = np.stack([smooth, smooth.copy()], axis=1)
    values[40:42, 1] += 1.2

    repaired = dmr_stage._repair_joint_spikes(values, dt)

    np.testing.assert_array_equal(repaired[:, 0], smooth)
    assert np.abs(repaired[:, 1] - smooth).max() < 0.1
    acceleration = np.abs(np.diff(repaired, n=2, axis=0)) / (dt * dt)
    assert acceleration.max() <= dmr_stage._ARM_SPIKE_MAX_ACCEL


@pytest.mark.mujoco
def test_arm_speed_limit_scales_each_arm_chain_along_its_ik_step() -> None:
    model = MujocoModel.from_robot("k1")
    groups = dmr_stage._joint_groups(model, K1_DMR_PROFILE)
    chains = [
        model.get_qpos_indices(chain)
        for chain in dmr_stage._arm_chains(model, K1_DMR_PROFILE, groups.body + groups.wrist)
    ]
    previous = np.array(model.q0, dtype=np.float64)
    target = previous.copy()
    # Near the K1 shoulder singularity, frame-local IK swung the shoulder by
    # more than a radian in one frame; the other arm here moves slowly.
    left_step = np.zeros(len(chains[0]))
    left_step[:3] = (1.2, 0.3, -0.6)
    target[chains[0]] += left_step
    target[chains[1][0]] += 0.1
    model.forward(target)

    dmr_stage._limit_arm_speed(model, previous, chains, 0.5)

    limited = model.get_qpos()
    np.testing.assert_allclose(limited[chains[0]] - previous[chains[0]], left_step * 0.5 / 1.2)
    np.testing.assert_allclose(limited[chains[1]], target[chains[1]])
    untouched = np.setdiff1d(np.arange(len(target)), np.concatenate(chains))
    np.testing.assert_array_equal(limited[untouched], target[untouched])


def test_default_orientation_smoothing_follows_turns_past_180_degrees() -> None:
    dt = 1.0 / 30.0
    yaw = np.radians(np.linspace(0.0, 360.0, 120))
    rotations = Rotation.from_euler("z", yaw[:, None]).as_matrix()

    smoothed = dmr_stage._smooth_rotation_sequence(
        rotations,
        smooth_time=0.05,
        dt=dt,
        mode=G1_DMR_PROFILE.orientation_smoothing_mode,
    )

    error = (Rotation.from_matrix(smoothed) * Rotation.from_matrix(rotations).inv()).magnitude()
    # Only the clamped clip ends lag a constant-rate turn; nothing flips at 180 degrees.
    assert np.degrees(error).max() < 2.0
    assert np.degrees(error[3:-3]).max() < 0.1


@pytest.mark.mujoco
def test_arm_chains_split_g1_shoulder_to_hand_joints_by_side() -> None:
    model = MujocoModel.from_robot("g1")
    groups = dmr_stage._joint_groups(model, G1_DMR_PROFILE)

    left, right = dmr_stage._arm_chains(model, G1_DMR_PROFILE, groups.body + groups.wrist)

    expected = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
    assert all(f"left_{joint}_joint" in left for joint in expected)
    assert all(f"right_{joint}_joint" in right for joint in expected)
    assert not set(left) & set(right)


@pytest.mark.mujoco
@pytest.mark.parametrize(
    ("robot_id", "profile", "left_joint"),
    (("g1", G1_DMR_PROFILE, "left_elbow_joint"), ("t1", T1_DMR_PROFILE, "left_elbow_yaw_joint")),
)
def test_elbow_axis_setup_finds_the_flexion_hinge(
    robot_id: str, profile: DmrProfile, left_joint: str
) -> None:
    motion = load_soma_motion(EXAMPLE)
    joi = extract_soma_joi(motion)
    source = tuple(dmr_stage._frame_transforms(joi, tick) for tick in range(motion.frame_count))
    model = MujocoModel.from_robot(robot_id)
    model.reset()

    setup = dmr_stage._elbow_axis_setup(model, profile, source)

    assert set(setup) == {"left", "right"}
    hinge_body, robot_axis, source_normals = (
        setup["left"].body,
        setup["left"].axis,
        setup["left"].normals,
    )
    hinge_joints = {
        model.model.joint(joint).name
        for joint in range(model.model.njnt)
        if model.body_names[int(model.model.jnt_bodyid[joint])] == hinge_body
    }
    assert left_joint in hinge_joints
    np.testing.assert_allclose(np.linalg.norm(robot_axis), 1.0)
    # SOMA elbows bend about the upper-arm y axis, even where the arm is straight.
    upper_arm_y = np.stack([dmr_stage.rotation(frame["ls"])[:, 1] for frame in source])
    np.testing.assert_allclose(np.linalg.norm(source_normals, axis=1), 1.0)
    alignment = np.clip(np.abs(np.sum(source_normals * upper_arm_y, axis=1)), 0.0, 1.0)
    assert np.degrees(np.arccos(alignment)).max() < 5.0
    np.testing.assert_array_equal(model.get_qpos(), model.q0)


@pytest.mark.mujoco
def test_arm_targets_do_not_move_the_trunk_or_legs() -> None:
    motion = load_soma_motion(EXAMPLE)
    frame_count = 3
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
    moved_joints = short_motion.posed_joints.copy()
    for name in ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand"):
        moved_joints[:, SOMA77_JOINT_NAMES.index(name)] += (0.05, 0.08, 0.05)
    moved_motion = replace(short_motion, posed_joints=moved_joints)

    base = dmr_stage.run_dmr(short_motion, robot_id="g1", backend="python").qpos
    moved = dmr_stage.run_dmr(moved_motion, robot_id="g1", backend="python").qpos

    model = MujocoModel.from_robot("g1")
    groups = dmr_stage._joint_groups(model, G1_DMR_PROFILE)
    arm_joints = dmr_stage._arm_joint_names(model, G1_DMR_PROFILE, groups.body)
    arm_columns = set(model.get_qpos_indices(arm_joints))
    arm_columns |= set(model.get_qpos_indices(groups.wrist))
    trunk_joint_columns = [
        column for column in range(7, base.shape[1]) if column not in arm_columns
    ]
    difference = np.abs(base.astype(np.float64) - moved.astype(np.float64))
    assert difference[:, sorted(arm_columns)].max() > 0.3
    # Solving arms in the whole-body primary moved these legs and waist by 0.14
    # rad and the root by 4 mm. What remains comes from the shoulder JOI body,
    # which sits behind the shoulder pitch joint.
    assert difference[:, trunk_joint_columns].max() < 0.02
    assert difference[:, :3].max() < 0.002


@pytest.mark.mujoco
@pytest.mark.parametrize("robot_id", ("g1", "h2", "apollo", "igrisc"))
def test_elbow_center_measures_the_hinge_bend(robot_id: str) -> None:
    # H2, Apollo and IGRIS-C elbow JOI bodies sit up to 10 deg off the bend the
    # hinge actually makes; the elbow centre must reproduce that bend exactly.
    profile = dmr_stage.get_dmr_profile(robot_id)
    motion = load_soma_motion(EXAMPLE)
    joi = extract_soma_joi(motion)
    source = tuple(dmr_stage._frame_transforms(joi, tick) for tick in range(motion.frame_count))
    model = MujocoModel.from_robot(robot_id)
    model.reset()
    elbow = dmr_stage._elbow_axis_setup(model, profile, source)["left"]
    neutral = model.get_qpos()
    body_id = model.body_names.index(elbow.body)
    joint = next(
        j
        for j in range(model.model.njnt)
        if model.model.jnt_bodyid[j] == body_id and model.model.jnt_type[j] == 3
    )
    lower, upper = model.model.jnt_range[joint]

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    for value in np.linspace(lower, upper, 7):
        trial = neutral.copy()
        trial[model.model.jnt_qposadr[joint]] = value
        model.forward(trial)
        shoulder, wrist = (
            dmr_stage.position(model.get_body_transform(profile.joi_bodies[key]))
            for key in ("ls", "lw")
        )
        center = elbow.center_position(model)
        anchor, axis = model.data.xanchor[joint], model.data.xaxis[joint]
        shoulder_offset, wrist_offset = shoulder - anchor, wrist - anchor
        hinge_bend = 180.0 - angle(
            shoulder_offset - np.dot(shoulder_offset, axis) * axis,
            wrist_offset - np.dot(wrist_offset, axis) * axis,
        )
        assert abs(angle(center - shoulder, wrist - center) - hinge_bend) < 1.0
        assert abs(np.linalg.norm(center - shoulder) - elbow.upper_length) < 0.01
        assert abs(np.linalg.norm(wrist - center) - elbow.fore_length) < 0.01
    model.forward(neutral)


def test_elbow_axis_target_is_enabled_by_default() -> None:
    assert G1_DMR_PROFILE.elbow_axis_weight == 2.0
    assert T1_DMR_PROFILE.elbow_axis_weight == 2.0


@pytest.mark.mujoco
@pytest.mark.parametrize("robot_id", ("oli", "t1", "pm01"))
def test_elbow_axis_follows_flexion_for_straight_neutral_elbows(robot_id: str) -> None:
    # These elbows are straight in the neutral pose and can bend both ways; the
    # axis must point along the bend normal on the larger (flexion) side.
    profile = dmr_stage.get_dmr_profile(robot_id)
    motion = load_soma_motion(EXAMPLE)
    joi = extract_soma_joi(motion)
    source = tuple(dmr_stage._frame_transforms(joi, tick) for tick in range(motion.frame_count))
    model = MujocoModel.from_robot(robot_id)
    model.reset()
    setup = dmr_stage._elbow_axis_setup(model, profile, source)
    neutral = model.get_qpos()

    def joi_flex_normal(qpos: np.ndarray, keys: tuple[str, ...]) -> tuple[np.ndarray, float]:
        model.forward(qpos)
        return dmr_stage._bend_normal(
            *(dmr_stage.position(model.get_body_transform(profile.joi_bodies[k])) for k in keys)
        )

    for side, keys in (("left", ("ls", "le", "lw")), ("right", ("rs", "re", "rw"))):
        hinge_body, robot_axis = setup[side].body, setup[side].axis
        body_id = model.body_names.index(hinge_body)
        joint = next(
            j
            for j in range(model.model.njnt)
            if model.model.jnt_bodyid[j] == body_id and model.model.jnt_type[j] == 3
        )
        address = model.model.jnt_qposadr[joint]
        lower, upper = model.model.jnt_range[joint]
        flexion_end = lower if neutral[address] - lower > upper - neutral[address] else upper
        trial = neutral.copy()
        trial[address] = neutral[address] + 0.5 * (flexion_end - neutral[address])
        normal, flex = joi_flex_normal(trial, keys)
        world_axis = dmr_stage.rotation(model.get_body_transform(hinge_body)) @ robot_axis
        assert flex > 20.0
        assert float(np.dot(normal, world_axis)) > 0.9
    model.forward(neutral)
