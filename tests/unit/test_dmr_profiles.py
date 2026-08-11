from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from core_retarget.exceptions import ConfigurationError
from core_retarget.mujoco import MujocoModel
from core_retarget.robots.joi import get_body_joi_mapping
from core_retarget.robots.profiles import DMR_PROFILES, get_dmr_profile
from core_retarget.robots.profiles.g1 import G1_DMR_PROFILE
from core_retarget.robots.profiles.h1 import H1_DMR_PROFILE
from core_retarget.robots.profiles.h2 import H2_DMR_PROFILE
from core_retarget.robots.profiles.k1 import K1_DMR_PROFILE
from core_retarget.robots.profiles.r1 import R1_DMR_PROFILE
from core_retarget.robots.profiles.schema import DmrProfile


class DmrProfileTest(unittest.TestCase):
    def test_profile_registry_covers_all_supported_robots(self) -> None:
        expected = (
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
        )
        self.assertEqual(tuple(DMR_PROFILES), expected)
        for robot_id in expected:
            with self.subTest(robot=robot_id):
                self.assertIs(get_dmr_profile(f" {robot_id.upper()} "), DMR_PROFILES[robot_id])

    def test_k1_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_dmr_profile(" K1 "), K1_DMR_PROFILE)

    def test_g1_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_dmr_profile(" G1 "), G1_DMR_PROFILE)

    def test_h1_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_dmr_profile(" H1 "), H1_DMR_PROFILE)

    def test_h2_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_dmr_profile(" H2 "), H2_DMR_PROFILE)

    def test_r1_lookup_is_case_insensitive(self) -> None:
        self.assertIs(get_dmr_profile(" R1 "), R1_DMR_PROFILE)

    def test_k1_profile_matches_verified_legacy_constants(self) -> None:
        profile = get_dmr_profile("k1")

        self.assertEqual(profile.qpos_dim, 30)
        self.assertEqual(profile.wrist_joint_tokens, ("wrist",))
        self.assertEqual(profile.link_length_base_reference, "legacy_midhip")

        self.assertEqual(profile.pelvis_orientation_reference_mode, "source_absolute")
        self.assertEqual(profile.pelvis_orientation_solve_stage, "legacy_post")
        self.assertEqual(profile.pelvis_orientation_weight, 0.03)
        self.assertEqual(profile.pelvis_orientation_axis_length, 0.15)
        self.assertEqual(profile.pelvis_orientation_smooth_time, 0.05)

        self.assertEqual(profile.ankle_orientation_mode, "source_body")
        self.assertEqual(profile.ankle_orientation_stage, "primary")
        self.assertEqual(profile.ankle_orientation_axes, (1, 2))
        self.assertEqual(profile.ankle_orientation_axis_length, 0.10)
        self.assertEqual(
            profile.left_ankle_local_offset,
            (
                (-0.12, 0.99, -0.02),
                (0.02, 0.02, 1.0),
                (0.99, 0.12, -0.02),
            ),
        )
        self.assertEqual(
            profile.right_ankle_local_offset,
            (
                (0.18, 0.98, 0.02),
                (0.05, -0.03, 1.0),
                (0.98, -0.18, -0.05),
            ),
        )

        self.assertEqual(
            profile.left_hand_local_offset,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(
            profile.right_hand_local_offset,
            ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(profile.left_hand_anchor_local, (0.03, 0.0, 0.0))
        self.assertEqual(profile.right_hand_anchor_local, (0.03, 0.0, 0.0))
        self.assertEqual(profile.left_hand_axis_signs, (1.0, -1.0, 1.0))
        self.assertEqual(profile.right_hand_axis_signs, (1.0, 1.0, 1.0))
        self.assertEqual(profile.hand_orientation_axis_length, 0.10)
        self.assertEqual(profile.initial_warmup_passes, 4)

        self.assertEqual(profile.body_solver.max_iterations, 100)
        self.assertEqual(profile.body_solver.revolute_step, 0.5)
        self.assertAlmostEqual(profile.body_solver.revolute_update_limit, math.radians(5.0))
        self.assertEqual(profile.body_solver.damping, 1e-2)
        self.assertAlmostEqual(profile.body_solver.joint_limit_probe, math.radians(3.0))

        self.assertEqual(profile.hand_solver.max_iterations, 10)
        self.assertEqual(profile.hand_solver.revolute_step, 0.5)
        self.assertAlmostEqual(profile.hand_solver.revolute_update_limit, math.radians(10.0))
        self.assertEqual(profile.hand_solver.damping, 1e-4)
        self.assertAlmostEqual(profile.hand_solver.joint_limit_probe, math.radians(3.0))

    def test_g1_profile_matches_verified_legacy_constants(self) -> None:
        profile = get_dmr_profile("g1")

        self.assertEqual(profile.qpos_dim, 36)
        self.assertEqual(profile.wrist_joint_tokens, ("wrist",))
        self.assertEqual(profile.waist_joint_tokens, ("waist_roll", "waist_pitch"))
        self.assertEqual(profile.ankle_joint_tokens, ("ankle",))
        self.assertTrue(profile.exclude_waist_from_primary_dmr)
        self.assertEqual(profile.link_length_base_reference, "legacy_midhip")

        self.assertEqual(profile.pelvis_orientation_reference_mode, "robot_neutral_delta")
        self.assertEqual(profile.pelvis_orientation_solve_stage, "primary")
        self.assertEqual(profile.pelvis_orientation_weight, 0.0)
        self.assertEqual(profile.pelvis_primary_orientation_weight, 0.01)
        self.assertEqual(profile.pelvis_primary_dynamic_orientation_weight, 0.3)
        self.assertEqual(profile.pelvis_orientation_axis_length, 0.25)
        self.assertEqual(profile.pelvis_orientation_smooth_time, 0.05)
        self.assertEqual(profile.pelvis_stabilization_strength, 0.85)
        self.assertEqual(profile.pelvis_stabilization_orientation_weight, 0.09)
        self.assertEqual(profile.pelvis_stabilization_linear_speed_low, 0.02)
        self.assertEqual(profile.pelvis_stabilization_linear_speed_high, 0.08)
        self.assertEqual(profile.pelvis_stabilization_angular_speed_low, 0.15)
        self.assertEqual(profile.pelvis_stabilization_angular_speed_high, 0.80)
        self.assertEqual(profile.pelvis_stabilization_smooth_time, 0.10)

        self.assertEqual(profile.trunk_position_mode, "robot_bind_local")
        self.assertEqual(profile.trunk_position_gate, "stability")
        self.assertEqual(profile.trunk_position_strength, 0.5)

        self.assertEqual(profile.torso_orientation_stage, "post")
        self.assertEqual(profile.torso_orientation_joi_key, "torso")
        self.assertEqual(profile.torso_orientation_reference_mode, "source_delta")
        self.assertEqual(profile.torso_orientation_weight, 0.03)
        self.assertEqual(profile.torso_orientation_axes, (2,))
        self.assertEqual(profile.torso_orientation_axis_length, 0.15)
        self.assertEqual(profile.torso_orientation_smooth_time, 0.05)

        self.assertEqual(profile.ankle_orientation_mode, "outsole_normal")
        self.assertEqual(profile.ankle_orientation_stage, "post")
        self.assertEqual(profile.ankle_orientation_axes, (2,))
        self.assertEqual(profile.ankle_orientation_axis_length, 0.15)
        self.assertEqual(profile.ankle_orientation_smooth_time, 0.10)
        self.assertIsNone(profile.left_ankle_orientation_joi_key)
        self.assertIsNone(profile.right_ankle_orientation_joi_key)

        self.assertIsNone(profile.left_hand_local_offset)
        self.assertIsNone(profile.right_hand_local_offset)
        self.assertEqual(profile.left_hand_anchor_local, (0.0, 0.0, 0.0))
        self.assertEqual(profile.right_hand_anchor_local, (0.0, 0.0, 0.0))
        self.assertEqual(profile.left_hand_axis_signs, (1.0, 1.0, 1.0))
        self.assertEqual(profile.right_hand_axis_signs, (1.0, 1.0, 1.0))
        self.assertEqual(profile.hand_orientation_reference_mode, "first_realized")
        self.assertEqual(profile.dmr_temporal_nullspace_gain, 0.25)

        self.assertEqual(
            profile.pelvis_stabilization_joint_smooth_tokens,
            ("hip", "knee", "ankle"),
        )
        self.assertEqual(profile.pelvis_stabilization_joint_median_window, 3)
        self.assertEqual(profile.pelvis_stabilization_joint_smooth_time, 0.08)
        self.assertAlmostEqual(
            profile.pelvis_stabilization_joint_smooth_max_delta,
            math.radians(6.0),
        )
        self.assertEqual(profile.pelvis_stabilization_joint_smooth_gate, "stability")

        assert profile.torso_solver is not None
        self.assertEqual(profile.torso_solver.max_iterations, 30)
        self.assertEqual(profile.torso_solver.revolute_step, 0.35)
        self.assertAlmostEqual(
            profile.torso_solver.revolute_update_limit,
            math.radians(2.0),
        )
        self.assertEqual(profile.torso_solver.damping, 1e-4)
        self.assertAlmostEqual(profile.torso_solver.joint_limit_probe, math.radians(2.0))

        assert profile.ankle_solver is not None
        self.assertEqual(profile.ankle_solver.max_iterations, 50)
        self.assertEqual(profile.ankle_solver.revolute_step, 0.5)
        self.assertAlmostEqual(
            profile.ankle_solver.revolute_update_limit,
            math.radians(2.0),
        )
        self.assertEqual(profile.ankle_solver.damping, 1e-4)
        self.assertAlmostEqual(profile.ankle_solver.joint_limit_probe, math.radians(3.0))

    def test_h1_profile_matches_verified_legacy_constants(self) -> None:
        profile = get_dmr_profile("h1")

        self.assertEqual(profile.qpos_dim, 27)
        self.assertEqual(profile.link_length_base_reference, "legacy_midhip")
        self.assertEqual(profile.pelvis_orientation_reference_mode, "source_absolute")
        self.assertEqual(profile.pelvis_orientation_solve_stage, "legacy_post")
        self.assertEqual(profile.pelvis_orientation_weight, 0.03)
        self.assertEqual(profile.pelvis_orientation_axis_length, 0.15)
        self.assertEqual(profile.pelvis_orientation_smooth_time, 0.05)

        self.assertEqual(profile.ankle_orientation_mode, "outsole_normal")
        self.assertEqual(profile.ankle_orientation_stage, "post")
        self.assertEqual(profile.ankle_orientation_axes, (2,))
        self.assertEqual(profile.ankle_orientation_axis_length, 0.15)
        self.assertEqual(profile.ankle_orientation_smooth_time, 0.10)
        self.assertEqual(profile.left_ankle_orientation_joi_key, "lsole")
        self.assertEqual(profile.right_ankle_orientation_joi_key, "rsole")
        self.assertEqual(
            profile.left_ankle_local_offset,
            (
                (-0.12, 0.99, -0.02),
                (0.02, 0.02, 1.0),
                (0.99, 0.12, -0.02),
            ),
        )
        self.assertEqual(
            profile.right_ankle_local_offset,
            (
                (0.18, 0.98, 0.02),
                (0.05, -0.03, 1.0),
                (0.98, -0.18, -0.05),
            ),
        )

        self.assertFalse(profile.hand_orientation_enabled)
        self.assertIsNone(profile.left_hand_local_offset)
        self.assertIsNone(profile.right_hand_local_offset)
        self.assertEqual(profile.hand_orientation_reference_mode, "first_realized")
        self.assertEqual(profile.initial_warmup_passes, 4)
        self.assertEqual(profile.pelvis_stabilization_strength, 0.0)
        self.assertEqual(profile.trunk_position_strength, 0.0)

        self.assertEqual(profile.body_solver.max_iterations, 100)
        self.assertAlmostEqual(profile.body_solver.revolute_update_limit, math.radians(5.0))
        assert profile.ankle_solver is not None
        self.assertEqual(profile.ankle_solver.max_iterations, 50)
        self.assertEqual(profile.ankle_solver.revolute_step, 0.5)
        self.assertAlmostEqual(
            profile.ankle_solver.revolute_update_limit,
            math.radians(2.0),
        )
        self.assertEqual(profile.ankle_solver.damping, 1e-4)
        self.assertAlmostEqual(profile.ankle_solver.joint_limit_probe, math.radians(3.0))

    def test_h2_profile_matches_the_inherited_g1_dmr_configuration(self) -> None:
        profile = get_dmr_profile("h2")

        self.assertEqual(profile.qpos_dim, 38)
        for field in fields(DmrProfile):
            if field.name in {"robot_id", "qpos_dim", "joi_bodies"}:
                continue
            with self.subTest(field=field.name):
                self.assertEqual(
                    getattr(profile, field.name),
                    getattr(G1_DMR_PROFILE, field.name),
                )

    def test_r1_profile_matches_the_research_g1_overrides(self) -> None:
        profile = get_dmr_profile("r1")
        overrides = {
            "robot_id",
            "joi_bodies",
            "joi_anchor_reference_keys",
            "torso_orientation_weight",
            "torso_orientation_stage",
            "torso_orientation_joi_key",
            "exclude_waist_from_primary_dmr",
            "trunk_position_mode",
            "trunk_position_gate",
            "trunk_position_strength",
        }

        self.assertEqual(profile.qpos_dim, 36)
        self.assertEqual(dict(profile.joi_anchor_reference_keys), {"base": ("lp", "rp")})
        self.assertEqual(profile.torso_orientation_weight, 0.0)
        self.assertEqual(profile.torso_orientation_stage, "none")
        self.assertEqual(profile.torso_orientation_joi_key, "spine")
        self.assertFalse(profile.exclude_waist_from_primary_dmr)
        self.assertEqual(profile.trunk_position_mode, "robot_neutral_delta")
        self.assertEqual(profile.trunk_position_gate, "always")
        self.assertEqual(profile.trunk_position_strength, 1.0)
        for field in fields(DmrProfile):
            if field.name in overrides:
                continue
            with self.subTest(field=field.name):
                self.assertEqual(
                    getattr(profile, field.name),
                    getattr(G1_DMR_PROFILE, field.name),
                )

    def test_new_profiles_match_research_capabilities(self) -> None:
        expected = {
            "apollo": (39, ("torso_roll", "torso_pitch"), ("wrist",), True, "post", True),
            "oli": (68, ("waist_roll", "waist_pitch"), ("wrist",), True, "post", True),
            "n1": (30, ("waist_yaw_joint",), ("wrist",), True, "post", True),
            "adam": (32, ("waistRoll", "waistPitch"), ("wrist",), True, "post", True),
            "t1": (30, ("Waist",), ("Elbow_Yaw",), True, "post", True),
            "pm01": (31, ("J12_WAIST_YAW",), ("ELBOW_YAW",), False, "none", False),
        }
        for robot_id, values in expected.items():
            with self.subTest(robot=robot_id):
                profile = get_dmr_profile(robot_id)
                qpos_dim, waist, wrist, hand_enabled, torso_stage, exclude_waist = values
                self.assertEqual(profile.qpos_dim, qpos_dim)
                self.assertEqual(profile.waist_joint_tokens, waist)
                self.assertEqual(profile.wrist_joint_tokens, wrist)
                self.assertEqual(profile.hand_orientation_enabled, hand_enabled)
                self.assertEqual(profile.torso_orientation_stage, torso_stage)
                self.assertEqual(profile.exclude_waist_from_primary_dmr, exclude_waist)
                self.assertEqual(
                    dict(profile.joi_bodies),
                    dict(get_body_joi_mapping(robot_id)),
                )
                self.assertEqual(
                    dict(profile.joi_anchor_reference_keys),
                    {"base": ("lp", "rp")},
                )
                self.assertEqual(profile.left_ankle_orientation_joi_key, "lsole")
                self.assertEqual(profile.right_ankle_orientation_joi_key, "rsole")

    def test_new_profile_joi_bodies_exist_in_packaged_models(self) -> None:
        for robot_id in ("apollo", "oli", "n1", "adam", "t1", "pm01"):
            with self.subTest(robot=robot_id):
                model = MujocoModel.from_robot(robot_id)
                for key, body_name in get_dmr_profile(robot_id).joi_bodies.items():
                    with self.subTest(robot=robot_id, joi=key):
                        transform = model.get_body_transform(body_name)
                        self.assertEqual(transform.shape, (4, 4))

    def test_profiles_are_deeply_immutable_and_copy_mappings(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            K1_DMR_PROFILE.qpos_dim = 31  # type: ignore[misc]
        with self.assertRaises(TypeError):
            K1_DMR_PROFILE.joi_bodies["base"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            K1_DMR_PROFILE.left_ankle_local_offset[0][0] = 0.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            R1_DMR_PROFILE.joi_anchor_reference_keys["base"] = ("rp",)  # type: ignore[index]

        external_mapping = dict(K1_DMR_PROFILE.joi_bodies)
        copied_profile = replace(K1_DMR_PROFILE, joi_bodies=external_mapping)
        external_mapping["base"] = "changed"
        self.assertEqual(copied_profile.joi_bodies["base"], "pelvis")

        external_anchors = {"base": ["lp", "rp"]}
        copied_profile = replace(
            R1_DMR_PROFILE,
            joi_anchor_reference_keys=external_anchors,  # type: ignore[arg-type]
        )
        external_anchors["base"].append("spine")
        self.assertEqual(copied_profile.joi_anchor_reference_keys["base"], ("lp", "rp"))

    def test_semantic_joi_anchor_metadata_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from joi_bodies"):
            replace(R1_DMR_PROFILE, joi_anchor_reference_keys={"unknown": ("lp",)})
        with self.assertRaisesRegex(ValueError, "must have reference keys"):
            replace(R1_DMR_PROFILE, joi_anchor_reference_keys={"base": ()})
        with self.assertRaisesRegex(ValueError, "unknown reference keys"):
            replace(R1_DMR_PROFILE, joi_anchor_reference_keys={"base": ("unknown",)})

    def test_k1_joi_mapping_matches_model_semantics(self) -> None:
        joi = get_dmr_profile("k1").joi_bodies
        self.assertEqual(joi["base"], "pelvis")
        self.assertEqual(joi["la"], "left_ankle_roll_link")
        self.assertEqual(joi["lf"], "left_ankle_roll_link")
        self.assertEqual(joi["ra"], "right_ankle_roll_link")
        self.assertEqual(joi["rf"], "right_ankle_roll_link")
        self.assertEqual(joi["lh"], "left_wrist")
        self.assertEqual(joi["rh"], "right_wrist")

    def test_g1_joi_mapping_matches_model_semantics(self) -> None:
        joi = get_dmr_profile("g1").joi_bodies
        self.assertEqual(joi["base"], "pelvis")
        self.assertEqual(joi["spine"], "waist_yaw_link")
        self.assertEqual(joi["torso"], "torso_link")
        self.assertEqual(joi["lf"], "left_ankle_roll_link")
        self.assertEqual(joi["rf"], "right_ankle_roll_link")
        self.assertEqual(joi["lh"], "left_wrist_yaw_link")
        self.assertEqual(joi["rh"], "right_wrist_yaw_link")

    def test_h1_joi_mapping_matches_model_semantics(self) -> None:
        joi = get_dmr_profile("h1").joi_bodies
        self.assertEqual(joi["base"], "pelvis")
        self.assertEqual(joi["lp"], "left_hip_pitch_link")
        self.assertEqual(joi["rp"], "right_hip_pitch_link")
        self.assertEqual(joi["spine"], "torso_link")
        self.assertEqual(joi["torso"], "torso_link")
        self.assertEqual(joi["lsole"], "left_sole_link")
        self.assertEqual(joi["rsole"], "right_sole_link")
        self.assertEqual(joi["lh"], "left_hand_link")
        self.assertEqual(joi["rh"], "right_hand_link")

    def test_h2_joi_mapping_matches_model_semantics(self) -> None:
        joi = get_dmr_profile("h2").joi_bodies
        self.assertEqual(dict(joi), dict(get_body_joi_mapping("h2")))
        self.assertEqual(joi["base"], "pelvis")
        self.assertEqual(joi["spine"], "waist_yaw_link")
        self.assertEqual(joi["torso"], "torso_link")
        self.assertEqual(joi["lf"], "left_ankle_pitch_link")
        self.assertEqual(joi["rf"], "right_ankle_pitch_link")
        self.assertEqual(joi["lsole"], "left_sole_link")
        self.assertEqual(joi["rsole"], "right_sole_link")
        self.assertEqual(joi["lh"], "left_wrist_yaw_link")
        self.assertEqual(joi["rh"], "right_wrist_yaw_link")

    def test_r1_joi_mapping_matches_model_semantics(self) -> None:
        joi = get_dmr_profile("r1").joi_bodies
        self.assertEqual(dict(joi), dict(get_body_joi_mapping("r1")))
        self.assertEqual(joi["base"], "pelvis")
        self.assertEqual(joi["lp"], "left_hip_roll_link_aux")
        self.assertEqual(joi["rp"], "right_hip_roll_link_aux")
        self.assertEqual(joi["spine"], "torso_link")
        self.assertEqual(joi["lw"], "left_wrist_roll_link")
        self.assertEqual(joi["rw"], "right_wrist_roll_link")

    def test_unknown_robot_uses_asset_registry_error(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Unsupported robot"):
            get_dmr_profile("not-a-robot")


if __name__ == "__main__":
    unittest.main()
