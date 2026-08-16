from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rimkit.exceptions import MotionValidationError
from rimkit.motion.soma import load_soma_motion, rotation_x, validate_soma_npz

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    REPOSITORY_ROOT / "examples" / "motions" / "kimodo" / "soma_rp_v11" / "stand_walk_run_stop.npz"
)


class SomaValidationTest(unittest.TestCase):
    def test_default_example_contract(self) -> None:
        summary = validate_soma_npz(EXAMPLE)
        self.assertEqual(summary.frame_count, 150)
        self.assertEqual(summary.fps, 30.0)
        self.assertEqual(summary.contact_channels, 6)
        self.assertEqual(
            summary.sha256,
            "16112abc72c0dbb85eb6b32d2ae284d40ebcc496214f9d5ac1fa8a29e12b9a07",
        )
        self.assertTrue(any("No fps field" in warning for warning in summary.warnings))

    def test_fps_override_has_priority(self) -> None:
        summary = validate_soma_npz(EXAMPLE, fps_override=60.0)
        self.assertEqual(summary.fps, 60.0)
        self.assertFalse(any("No fps field" in warning for warning in summary.warnings))

    def test_all_execution_examples_load_in_notebook_world_frame(self) -> None:
        examples = (
            ("alternating_lunges_contacts.npz", 270),
            ("backward_walk_contacts.npz", 240),
            ("foot_walk_stop.npz", 240),
            ("jump_land_contacts.npz", 195),
            ("side_steps_right_contacts.npz", 270),
            ("slow_walk_firm_steps.npz", 270),
            ("stand_walk_run_stop.npz", 150),
            ("march_in_place_contacts.npz", 240),
        )
        for filename, expected_frames in examples:
            with self.subTest(filename=filename):
                path = EXAMPLE.with_name(filename)
                motion = load_soma_motion(path)
                with np.load(path, allow_pickle=False) as archive:
                    source_positions = np.asarray(archive["posed_joints"], dtype=np.float64)
                    source_rotations = np.asarray(archive["global_rot_mats"], dtype=np.float64)
                    source_contacts = np.asarray(archive["foot_contacts"])

                z_up = rotation_x(np.pi / 2.0)
                expected_positions = np.einsum("ij,taj->tai", z_up, source_positions)
                expected_rotations = np.einsum("ij,tajk->taik", z_up, source_rotations)

                self.assertEqual(motion.frame_count, expected_frames)
                self.assertEqual(motion.fps, 30.0)
                self.assertEqual(motion.posed_joints.dtype, np.float64)
                self.assertEqual(motion.global_rot_mats.dtype, np.float64)
                np.testing.assert_array_equal(motion.posed_joints, expected_positions)
                np.testing.assert_array_equal(motion.global_rot_mats, expected_rotations)
                np.testing.assert_array_equal(
                    motion.seconds,
                    np.arange(expected_frames, dtype=np.float64) / 30.0,
                )
                np.testing.assert_array_equal(motion.foot_contacts, source_contacts)
                self.assertFalse(motion.posed_joints.flags.writeable)
                self.assertFalse(motion.global_rot_mats.flags.writeable)
                self.assertFalse(motion.seconds.flags.writeable)
                self.assertIsNotNone(motion.foot_contacts)
                assert motion.foot_contacts is not None
                self.assertFalse(motion.foot_contacts.flags.writeable)

    def test_missing_required_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, posed_joints=np.zeros((2, 77, 3), dtype=np.float32))
            with self.assertRaises(MotionValidationError):
                validate_soma_npz(path)

    def test_object_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            np.savez(
                path,
                posed_joints=np.zeros((2, 77, 3), dtype=np.float32),
                global_rot_mats=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 77, 3, 3)),
                metadata=np.array([{"unsafe": True}], dtype=object),
            )
            with self.assertRaises(MotionValidationError):
                validate_soma_npz(path)

    def test_complex_required_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complex.npz"
            np.savez(
                path,
                posed_joints=np.zeros((2, 77, 3), dtype=np.complex128),
                global_rot_mats=np.broadcast_to(np.eye(3, dtype=np.float64), (2, 77, 3, 3)),
            )
            with self.assertRaises(MotionValidationError):
                validate_soma_npz(path)

    def test_non_numeric_optional_array_is_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "string-root.npz"
            np.savez(
                path,
                posed_joints=np.zeros((2, 77, 3), dtype=np.float64),
                global_rot_mats=np.broadcast_to(np.eye(3, dtype=np.float64), (2, 77, 3, 3)),
                root_positions=np.full((2, 3), "invalid"),
            )
            with self.assertRaises(MotionValidationError):
                validate_soma_npz(path)


if __name__ == "__main__":
    unittest.main()
