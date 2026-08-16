from __future__ import annotations

import unittest

import numpy as np

from rimkit.mujoco.model import MujocoModel
from rimkit.robots.registry import list_robots
from rimkit.robots.validation import verify_robot


class MuJoCoModelContractTest(unittest.TestCase):
    def test_all_public_scenes_compile(self) -> None:
        for robot in list_robots():
            with self.subTest(robot=robot.robot_id):
                result = verify_robot(robot, load_mujoco=True)
                self.assertTrue(result.ok, result.issues)
                self.assertEqual(result.model_info["nq"], robot.expected_nq)
                self.assertEqual(result.model_info["nv"], robot.expected_nv)
                self.assertEqual(result.model_info["nu"], robot.expected_nu)

    def test_all_public_scenes_share_lighting_contract(self) -> None:
        reference = MujocoModel.from_robot("g1").model
        light_fields = (
            "light_active",
            "light_ambient",
            "light_attenuation",
            "light_bodyid",
            "light_bulbradius",
            "light_castshadow",
            "light_cutoff",
            "light_diffuse",
            "light_dir",
            "light_exponent",
            "light_intensity",
            "light_mode",
            "light_pos",
            "light_range",
            "light_specular",
            "light_targetbodyid",
            "light_texid",
            "light_type",
        )

        for robot in list_robots():
            with self.subTest(robot=robot.robot_id):
                model = MujocoModel.from_robot(robot.robot_id).model
                self.assertEqual(int(model.nlight), int(reference.nlight))
                for field in light_fields:
                    np.testing.assert_array_equal(
                        getattr(model, field),
                        getattr(reference, field),
                        err_msg=f"{robot.robot_id}: {field}",
                    )

                self.assertEqual(model.vis.headlight.active, reference.vis.headlight.active)
                for field in ("ambient", "diffuse", "specular"):
                    np.testing.assert_array_equal(
                        getattr(model.vis.headlight, field),
                        getattr(reference.vis.headlight, field),
                        err_msg=f"{robot.robot_id}: headlight.{field}",
                    )


if __name__ == "__main__":
    unittest.main()
