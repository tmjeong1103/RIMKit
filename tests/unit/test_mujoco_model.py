from __future__ import annotations

import unittest

import numpy as np

from rimkit.mujoco import MujocoModel
from rimkit.robots.registry import list_robots

K1_JOINT_NAMES = (
    "root",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


class MujocoModelTest(unittest.TestCase):
    def test_all_registered_scenes_load_through_independent_adapter(self) -> None:
        for spec in list_robots():
            with self.subTest(robot=spec.robot_id):
                robot = MujocoModel.from_robot(spec.robot_id)
                self.assertEqual(robot.model.nq, spec.expected_nq)
                self.assertEqual(robot.model.nv, spec.expected_nv)
                self.assertEqual(robot.model.nu, spec.expected_nu)

    def test_k1_contract_home_and_joint_order(self) -> None:
        robot = MujocoModel.from_robot("k1")

        self.assertEqual((robot.model.nq, robot.model.nv, robot.model.nu), (30, 29, 23))
        self.assertEqual(robot.joint_names, K1_JOINT_NAMES)
        self.assertEqual(robot.rev_joint_names, K1_JOINT_NAMES[1:])
        self.assertEqual(robot.pri_joint_names, ())
        self.assertEqual(robot.rev_pri_joint_names, K1_JOINT_NAMES[1:])

        expected_q0 = np.zeros(30, dtype=np.float64)
        expected_q0[2] = 0.7955
        expected_q0[3] = 1.0
        np.testing.assert_array_equal(robot.q0, expected_q0)
        np.testing.assert_array_equal(robot.get_qpos(), expected_q0)
        self.assertFalse(robot.q0.flags.writeable)

    def test_indices_and_selected_qpos_expand_multidof_joints(self) -> None:
        robot = MujocoModel.from_robot("k1")

        np.testing.assert_array_equal(
            robot.get_qpos_indices(("root", "left_hip_pitch_joint", "right_wrist_roll_joint")),
            np.array([0, 7, 29], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            robot.get_dof_indices(("root", "left_hip_pitch_joint", "right_wrist_roll_joint")),
            np.array([0, 1, 2, 3, 4, 5, 6, 28], dtype=np.int32),
        )
        np.testing.assert_array_equal(robot.get_qpos("root"), robot.q0[:7])

    def test_forward_clips_joints_and_returns_world_transform(self) -> None:
        robot = MujocoModel.from_robot("k1")
        qpos = robot.q0.copy()
        qpos[:3] = (0.2, -0.1, 1.0)
        qpos[7] = 100.0
        robot.forward(qpos)

        self.assertAlmostEqual(float(robot.data.qpos[7]), 2.4435)
        pelvis = robot.get_body_transform("pelvis")
        np.testing.assert_allclose(pelvis[:3, 3], (0.2, -0.1, 1.0), atol=1e-12)
        np.testing.assert_allclose(pelvis[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(pelvis[3], (0.0, 0.0, 0.0, 1.0))

        robot.reset()
        np.testing.assert_array_equal(robot.get_qpos(), robot.q0)

    def test_state_copy_synchronizes_kinematics(self) -> None:
        source = MujocoModel.from_robot("k1")
        destination = MujocoModel.from_robot("k1")
        qpos = source.q0.copy()
        qpos[:3] = (0.1, 0.2, 0.9)
        qpos[10] = 0.4
        source.forward(qpos)
        source.data.qvel[:] = np.linspace(0.0, 0.28, source.model.nv)
        source.data.ctrl[:] = np.linspace(-0.1, 0.1, source.model.nu)
        source.data.time = 1.25

        destination.copy_state_from(source)

        np.testing.assert_array_equal(destination.data.qpos, source.data.qpos)
        np.testing.assert_array_equal(destination.data.qvel, source.data.qvel)
        np.testing.assert_array_equal(destination.data.ctrl, source.data.ctrl)
        self.assertEqual(float(destination.data.time), 1.25)
        np.testing.assert_allclose(
            destination.get_body_transform("left_ankle_roll_link"),
            source.get_body_transform("left_ankle_roll_link"),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
