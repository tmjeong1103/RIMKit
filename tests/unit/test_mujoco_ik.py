from __future__ import annotations

import unittest

import mujoco
import numpy as np

from core_retarget.mujoco.ik import BodyPositionIKSolver

_MODEL_XML = """
<mujoco model="one-link-ik">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.02"/>
      <body name="tip" pos="1 0 0">
        <geom type="sphere" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _TinyAdapter:
    rev_joint_names = ("hinge",)
    pri_joint_names: tuple[str, ...] = ()
    rev_pri_joint_names = ("hinge",)

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def copy_state_from(self, other: _TinyAdapter) -> None:
        self.data.qpos[:] = other.data.qpos
        self.data.qvel[:] = other.data.qvel
        mujoco.mj_forward(self.model, self.data)


class BodyPositionIKSolverTest(unittest.TestCase):
    def test_point_target_improves_and_keeps_candidate_history(self) -> None:
        adapter = _TinyAdapter()
        solver = BodyPositionIKSolver(
            adapter,
            max_iterations=12,
            revolute_step=0.5,
            revolute_update_limit=0.1,
            damping=1e-4,
            joint_limit_probe=0.02,
        )
        tip_id = adapter.model.body("tip").id
        current = adapter.data.xpos[tip_id].copy()
        desired_angle = 0.35
        target = np.asarray([np.cos(desired_angle), np.sin(desired_angle), 0.0])
        solver.add_target("tip", current, target)

        result = solver.solve()

        self.assertLess(result.error, result.errors[0] * 0.05)
        self.assertEqual(result.iterations, 12)
        self.assertEqual(result.errors.shape, (13,))
        self.assertAlmostEqual(result.joint_qpos[0], desired_angle, places=3)

    def test_joint_limit_probe_prevents_outward_motion(self) -> None:
        adapter = _TinyAdapter()
        adapter.data.qpos[0] = 0.49
        mujoco.mj_forward(adapter.model, adapter.data)
        solver = BodyPositionIKSolver(
            adapter,
            max_iterations=5,
            revolute_step=1.0,
            revolute_update_limit=0.2,
            damping=1e-5,
            joint_limit_probe=0.03,
        )
        tip_id = adapter.model.body("tip").id
        current = adapter.data.xpos[tip_id].copy()
        target = np.asarray([np.cos(1.0), np.sin(1.0), 0.0])
        solver.add_target("tip", current, target)

        result = solver.solve(joint_limits=True)

        self.assertLessEqual(result.joint_qpos[0], 0.5)
        self.assertAlmostEqual(result.joint_qpos[0], 0.49, places=12)

    def test_solve_syncs_state_without_clearing_targets(self) -> None:
        source = _TinyAdapter()
        internal = _TinyAdapter()
        solver = BodyPositionIKSolver(internal, max_iterations=0)
        tip_id = internal.model.body("tip").id
        current = internal.data.xpos[tip_id].copy()
        solver.add_target("tip", current, current)
        source.data.qpos[0] = 0.2
        mujoco.mj_forward(source.model, source.data)

        result = solver.solve(source_model=source)

        self.assertEqual(solver.target_count, 1)
        self.assertAlmostEqual(result.joint_qpos[0], 0.2, places=12)

    def test_configure_nullspace_copies_and_validates_reference(self) -> None:
        adapter = _TinyAdapter()
        solver = BodyPositionIKSolver(adapter)
        home = np.asarray([0.2], dtype=np.float64)

        solver.configure_nullspace(home, gain=0.25)
        home[0] = -0.4

        self.assertEqual(solver.nullspace_gain, 0.25)
        self.assertIsNotNone(solver._home)
        assert solver._home is not None
        self.assertEqual(solver._home[0], 0.2)

        for invalid_home in ([0.0, 0.1], [np.nan]):
            with self.subTest(home=invalid_home):
                with self.assertRaises(ValueError):
                    solver.configure_nullspace(invalid_home, gain=0.25)
        with self.assertRaises(ValueError):
            solver.configure_nullspace([0.0], gain=-0.1)


if __name__ == "__main__":
    unittest.main()
