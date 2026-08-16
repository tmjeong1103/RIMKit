from __future__ import annotations

import unittest

from rimkit.exceptions import ConfigurationError
from rimkit.robots.registry import get_robot, list_robots
from rimkit.robots.validation import verify_robot


class RobotRegistryTest(unittest.TestCase):
    def test_registry_has_exact_public_scope(self) -> None:
        self.assertEqual(
            [robot.robot_id for robot in list_robots()],
            [
                "g1",
                "h1",
                "h2",
                "r1",
                "k1",
                "apollo",
                "oli",
                "n1",
                "adam",
                "t1",
                "pm01",
            ],
        )

    def test_lookup_is_case_insensitive(self) -> None:
        self.assertEqual(get_robot(" G1 ").robot_id, "g1")

    def test_unknown_robot_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            get_robot("unsupported")

    def test_static_asset_contracts(self) -> None:
        for robot in list_robots():
            with self.subTest(robot=robot.robot_id):
                result = verify_robot(robot, load_mujoco=False)
                self.assertTrue(result.ok, result.issues)


if __name__ == "__main__":
    unittest.main()
