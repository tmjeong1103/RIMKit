from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core_retarget.motion import (
    SOMA77_JOINT_INDEX,
    SOMA77_JOINT_NAMES,
    SOMA77_JOINT_PARENTS,
    SOMA_JOI_NAMES,
    SOMA_PELVIS_LOCAL_ALIGNMENT,
    canonical_soma_pelvis_rotations,
    extract_soma_joi,
    load_soma_motion,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


class SomaJointsTest(unittest.TestCase):
    def _synthetic_motion(self, directory: str, *, frames: int = 2):
        positions = np.empty((frames, 77, 3), dtype=np.float64)
        rotations = np.empty((frames, 77, 3, 3), dtype=np.float64)
        for frame in range(frames):
            for joint in range(77):
                positions[frame, joint] = (
                    1000.0 * frame + joint,
                    100.0 + joint,
                    200.0 - joint,
                )
                rotations[frame, joint] = _rotation_z(0.001 * (frame + joint))
        path = Path(directory) / "synthetic.npz"
        np.savez(path, posed_joints=positions, global_rot_mats=rotations)
        return load_soma_motion(path, z_up=False), positions, rotations

    def test_topology_has_the_legacy_soma77_order(self) -> None:
        self.assertEqual(len(SOMA77_JOINT_PARENTS), 77)
        self.assertEqual(len(SOMA77_JOINT_NAMES), 77)
        self.assertEqual(SOMA77_JOINT_PARENTS[0], ("Hips", None))
        self.assertEqual(SOMA77_JOINT_INDEX["Spine2"], 2)
        self.assertEqual(SOMA77_JOINT_INDEX["LeftShoulder"], 11)
        self.assertEqual(SOMA77_JOINT_INDEX["RightShoulder"], 39)
        self.assertEqual(SOMA77_JOINT_INDEX["LeftLeg"], 67)
        self.assertEqual(SOMA77_JOINT_INDEX["LeftToeBase"], 70)
        self.assertEqual(SOMA77_JOINT_INDEX["RightLeg"], 72)
        self.assertEqual(SOMA77_JOINT_INDEX["RightToeBase"], 75)
        self.assertEqual(SOMA77_JOINT_PARENTS[-1], ("RightToeEnd", "RightToeBase"))

    def test_all_legacy_joi_mappings_and_composite_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            motion, positions, rotations = self._synthetic_motion(directory)
            joi = extract_soma_joi(motion)

        self.assertEqual(joi.names, SOMA_JOI_NAMES)
        self.assertEqual(joi.transforms.shape, (2, 22, 4, 4))
        expected_base = 0.5 * (
            positions[:, SOMA77_JOINT_INDEX["RightLeg"]]
            + positions[:, SOMA77_JOINT_INDEX["LeftLeg"]]
        )
        expected_neck = 0.5 * (
            positions[:, SOMA77_JOINT_INDEX["RightShoulder"]]
            + positions[:, SOMA77_JOINT_INDEX["LeftShoulder"]]
        )
        np.testing.assert_array_equal(joi.positions("base"), expected_base)
        np.testing.assert_array_equal(joi.rotations("base"), rotations[:, 0])
        np.testing.assert_array_equal(joi.positions("spine"), positions[:, 2])
        np.testing.assert_array_equal(joi.positions("neck"), expected_neck)
        np.testing.assert_array_equal(
            joi.rotations("neck"), rotations[:, SOMA77_JOINT_INDEX["Neck1"]]
        )
        np.testing.assert_array_equal(
            joi.positions("rs"), positions[:, SOMA77_JOINT_INDEX["RightArm"]]
        )
        np.testing.assert_array_equal(joi["rw"], joi["rh"])
        np.testing.assert_array_equal(joi["lw"], joi["lh"])
        np.testing.assert_array_equal(joi["ra"], joi["rf"])
        np.testing.assert_array_equal(joi["la"], joi["lf"])
        np.testing.assert_array_equal(
            joi.positions("rtoe"), positions[:, SOMA77_JOINT_INDEX["RightToeBase"]]
        )
        np.testing.assert_array_equal(
            joi.positions("ltoe"), positions[:, SOMA77_JOINT_INDEX["LeftToeBase"]]
        )
        self.assertFalse(joi.transforms.flags.writeable)
        self.assertFalse(joi.seconds.flags.writeable)

    def test_base_between_pelvis_false_preserves_hips_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            motion, positions, _ = self._synthetic_motion(directory)
            joi = extract_soma_joi(motion, base_between_pelvis=False)

        np.testing.assert_array_equal(joi.positions("base"), positions[:, 0])

    def test_source_stance_vector_is_not_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            motion, positions, _ = self._synthetic_motion(directory)
            joi = extract_soma_joi(motion)

        expected = (
            positions[:, SOMA77_JOINT_INDEX["LeftToeBase"]]
            - positions[:, SOMA77_JOINT_INDEX["RightToeBase"]]
        )
        np.testing.assert_array_equal(joi.positions("ltoe") - joi.positions("rtoe"), expected)

    def test_canonical_pelvis_is_a_right_local_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            motion, _, rotations = self._synthetic_motion(directory)
            canonical = canonical_soma_pelvis_rotations(motion)

        expected = rotations[:, SOMA77_JOINT_INDEX["Hips"]] @ SOMA_PELVIS_LOCAL_ALIGNMENT
        np.testing.assert_array_equal(canonical, expected)
        self.assertFalse(canonical.flags.writeable)


if __name__ == "__main__":
    unittest.main()
