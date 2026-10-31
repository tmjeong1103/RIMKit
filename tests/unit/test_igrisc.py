from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mujoco
import numpy as np
import pytest

from rimkit.motion import load_soma_motion
from rimkit.motion.contacts import ContactSchedule
from rimkit.mujoco.ground import foot_ground_signed_distance
from rimkit.mujoco.ik import BodyPositionIKSolver
from rimkit.mujoco.model import MujocoModel
from rimkit.native import resolve_backend
from rimkit.robots.profiles import get_dmr_profile, get_fpa_profile
from rimkit.stages.ara import run_ara
from rimkit.stages.dmr import run_dmr
from rimkit.stages.fpa import (
    FpaTargetsResult,
    _apply_root_z_correction,
    _fpa_bodies,
    _fpa_joint_groups,
    _ground_distances,
    build_fpa_targets,
)
from rimkit.stages.target_trajectories import run_target_trajectories


def test_igrisc_base_trajectory_stays_fixed_under_hip_articulation() -> None:
    model = MujocoModel.from_robot("igrisc")
    profile = get_dmr_profile("igrisc")
    poses = np.repeat(model.q0[None, :], 2, axis=0)
    poses[1, model.get_qpos_indices("Joint_Hip_Pitch_Left")] = 0.4
    base_positions = []
    hip_midpoints = []
    for pose in poses:
        model.forward(pose)
        base_positions.append(model.get_body_transform(profile.joi_bodies["base"])[:3, 3])
        hip_midpoints.append(
            0.5
            * (
                model.get_body_transform(profile.joi_bodies["lp"])[:3, 3]
                + model.get_body_transform(profile.joi_bodies["rp"])[:3, 3]
            )
        )

    result = run_target_trajectories(poses, poses, np.arange(2) / 30.0, robot_id="igrisc", fps=30.0)
    np.testing.assert_allclose(result.root, base_positions, atol=1e-12)
    np.testing.assert_allclose(result.root_smoothed, base_positions, atol=1e-12)
    np.testing.assert_allclose(result.root[0], result.root[1], atol=1e-12)
    assert np.linalg.norm(hip_midpoints[1] - hip_midpoints[0]) > 0.001


def test_igrisc_ground_queries_use_physical_feet_and_keep_aux_position_tasks() -> None:
    model = MujocoModel.from_robot("igrisc")
    poses = model.q0[None, :].copy()
    poses[:, 2] += 1.0
    bodies = _fpa_bodies("igrisc")
    distances = _ground_distances(model, bodies, poses)
    for side, distance in zip(("Right", "Left"), distances, strict=True):
        expected = foot_ground_signed_distance(
            model, poses, foot_body_name=f"Link_Ankle_Roll_{side}"
        )
        np.testing.assert_array_equal(distance, expected)
        assert np.isfinite(distance).all()
    assert bodies.right_foot.endswith("_aux")
    assert bodies.left_foot.endswith("_aux")
    groups = _fpa_joint_groups(model, get_fpa_profile("igrisc"))
    assert len(groups.left_recovery) == len(groups.right_recovery) == 6
    assert not any("waist" in name.lower() for name in groups.all)


def test_igrisc_gemx_root_z_correction_limits_support_transfer_steps() -> None:
    model = MujocoModel.from_robot("igrisc")
    poses = np.repeat(model.q0[None, :], 6, axis=0)
    poses[:, 2] = [0.8, 0.8, 0.8, 0.88, 0.88, 0.88]
    targets = cast(FpaTargetsResult, SimpleNamespace(right_toe=np.zeros((6, 3))))
    bodies = _fpa_bodies("igrisc")
    raw = _apply_root_z_correction(model, bodies, poses.copy(), targets, np.ones(6), np.zeros(6))
    limited_poses = poses.copy()
    profile = get_fpa_profile("igrisc", source_provider="gem-x")
    limited = _apply_root_z_correction(
        model,
        bodies,
        limited_poses,
        targets,
        np.ones(6),
        np.zeros(6),
        max_step=profile.root_z_correction_max_step,
    )
    assert np.max(np.abs(np.diff(raw))) == pytest.approx(0.08)
    assert np.max(np.abs(np.diff(limited))) <= 0.02 + 1e-12
    np.testing.assert_allclose(limited[:3], raw[:3], atol=1e-12)
    np.testing.assert_array_equal(limited_poses[:, 3:], poses[:, 3:])


@pytest.mark.parametrize("provider", ("kimodo", "gem-x"))
def test_igrisc_fpa_keeps_ara_base_and_merged_contact_floor_targets(provider: str) -> None:
    model = MujocoModel.from_robot("igrisc")
    profile = get_dmr_profile("igrisc")
    poses = np.repeat(model.q0[None, :], 8, axis=0)
    poses[:, 2] += 0.9
    poses[:, model.get_qpos_indices("Joint_Hip_Pitch_Left")] = np.linspace(0.0, 0.4, 8)[:, None]
    seconds = np.arange(8) / 30.0
    trajectories = run_target_trajectories(poses, poses, seconds, robot_id="igrisc", fps=30.0)
    hard_contact = np.array([True] * 7 + [False])
    contacts = cast(
        ContactSchedule,
        SimpleNamespace(
            frame_count=8,
            seconds=seconds,
            fps=30.0,
            right_contact_segments=np.array([[0, 7]]),
            left_contact_segments=np.array([[0, 7]]),
            right_contact_label=hard_contact,
            left_contact_label=hard_contact,
            right_confidence=np.full(8, 1.0 / 3.0),
            left_confidence=np.full(8, 1.0 / 3.0),
        ),
    )
    ara = run_ara(trajectories, contacts, robot_id="igrisc")
    targets = build_fpa_targets(
        poses, trajectories, ara, contacts, robot_id="igrisc", fps=30.0, source_provider=provider
    )
    for pose, expected in zip(targets.qpos_ara, ara.root_ara, strict=True):
        model.forward(pose)
        np.testing.assert_allclose(
            model.get_body_transform(profile.joi_bodies["base"])[:3, 3], expected, atol=1e-12
        )
    np.testing.assert_array_equal(targets.qpos_ara[:, 3:], poses[:, 3:])
    for weights, toe, clearance, ara_toe in (
        (
            targets.right_floor_weight,
            targets.right_toe_reference,
            targets.right_sole_clearance,
            ara.right_toe_ara,
        ),
        (
            targets.left_floor_weight,
            targets.left_toe_reference,
            targets.left_sole_clearance,
            ara.left_toe_ara,
        ),
    ):
        np.testing.assert_array_equal(weights[:7], 1.0 if provider == "gem-x" else 1.0 / 3.0)
        assert weights[-1] == pytest.approx(1.0 / 3.0)
        expected_z = (1.0 - weights) * ara_toe[:, 2] + weights * (
            ara.toe_floor_target_z + clearance
        )
        np.testing.assert_allclose(toe[:7, 2], expected_z[:7], atol=1e-12)
        if provider == "gem-x":
            np.testing.assert_allclose(np.ptp(toe[:7, :2], axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("provider", ("kimodo", "gem-x"))
@pytest.mark.parametrize("side,key", (("Left", "la_rot"), ("Right", "ra_rot")))
def test_igrisc_rotation_frames_keep_ankle_origins_stationary(
    provider: str, side: str, key: str
) -> None:
    model = MujocoModel.from_robot("igrisc")
    profile = get_dmr_profile("igrisc", source_provider=provider)
    selected_key = (
        profile.left_ankle_orientation_joi_key
        if side == "Left"
        else profile.right_ankle_orientation_joi_key
    )
    assert selected_key == key
    body = profile.joi_bodies[key]
    dofs = model.get_dof_indices((f"Joint_Ankle_Pitch_{side}", f"Joint_Ankle_Roll_{side}"))
    jacobian = np.empty((3, model.model.nv))
    mujoco.mj_jacBody(model.model, model.data, jacobian, None, model.model.body(body).id)
    np.testing.assert_allclose(jacobian[:, dofs], 0.0, atol=1e-12)
    sole_key = "lsole" if side == "Left" else "rsole"
    np.testing.assert_allclose(
        model.get_body_transform(body)[:3, :3],
        model.get_body_transform(profile.joi_bodies[sole_key])[:3, :3],
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ("python", "native"))
def test_igrisc_stationary_ankle_post_ik_converges_without_alternating(backend: str) -> None:
    if backend == "native" and not resolve_backend("auto").is_native:
        pytest.skip("native extension is not installed")
    model = MujocoModel.from_robot("igrisc")
    profile = get_dmr_profile("igrisc")
    joints = tuple(name for name in model.rev_joint_names if "Ankle" in name)
    columns = model.get_qpos_indices(joints)
    pose = model.q0.copy()
    pose[columns] = np.deg2rad(1.0)
    model.forward(pose)
    solver = BodyPositionIKSolver(
        model,
        max_iterations=50,
        revolute_step=0.5,
        revolute_update_limit=np.deg2rad(2.0),
        damping=1e-4,
        joint_limit_probe=np.deg2rad(3.0),
        nullspace_gain=0.0,
        backend=backend,
    )
    solved = []
    for _ in range(8):
        solver.reset_targets()
        for key in ("la_rot", "ra_rot"):
            body = profile.joi_bodies[key]
            frame = model.get_body_transform(body)
            solver.add_target(body, frame[:3, 3] + 0.10 * frame[:3, 2], frame[:3, 3] + (0, 0, 0.10))
        result = solver.solve(joints=joints)
        model.forward(result.qpos)
        solved.append(result.qpos[columns])
        fixed = np.ones(model.model.nq, dtype=bool)
        fixed[columns] = False
        np.testing.assert_array_equal(result.qpos[fixed], pose[fixed])
    assert np.max(np.abs(solved)) < np.deg2rad(0.01)


def test_gemx_frame_zero_nullspace_guidance_does_not_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz"
    )
    motion = load_soma_motion(source)
    motion = replace(
        motion,
        summary=replace(motion.summary, frame_count=2, duration_seconds=1.0 / motion.fps),
        seconds=motion.seconds[:2],
        posed_joints=motion.posed_joints[:2],
        global_rot_mats=motion.global_rot_mats[:2],
        foot_contacts=None,
    )
    gains: dict[int, list[float]] = {}
    configure = BodyPositionIKSolver.configure_nullspace

    def record(solver, home, *, gain):
        gains.setdefault(id(solver), []).append(gain)
        configure(solver, home, gain=gain)

    monkeypatch.setattr(BodyPositionIKSolver, "configure_nullspace", record)
    run_dmr(
        motion,
        robot_id="igrisc",
        source_provider="gem-x",
        left_contact_confidence=np.ones(2),
        right_contact_confidence=np.ones(2),
        backend="python",
    )
    # The body solver takes the frame-zero gain on frame zero only; the arm
    # solver never takes it.
    assert sorted(gains.values()) == [[0.0, 0.0], [1.0, 0.0]]
