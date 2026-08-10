from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core_retarget.exceptions import ArtifactError, MotionValidationError
from core_retarget.export.motion import (
    CONTACT_LABEL_NAMES,
    QPOS_LAYOUT,
    ROBOT_MOTION_FORMAT,
    ROBOT_MOTION_SCHEMA_VERSION,
    ROOT_QPOS_NAMES,
    build_robot_motion_arrays,
    write_robot_motion_npz,
)

SOURCE_SHA256 = "a" * 64


def _qpos(frame_count: int = 4, qpos_dim: int = 36) -> np.ndarray:
    values = np.zeros((frame_count, qpos_dim), dtype=np.float64)
    values[:, 3] = 1.0
    return values


def _contacts(frame_count: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros((frame_count, 4), dtype=bool)
    labels[:2, 0] = True
    labels[1:3, 1] = True
    confidence = labels.astype(np.float64)
    availability = np.asarray([True, True, False, False])
    return labels, confidence, availability


def _build_payload() -> dict[str, np.ndarray]:
    labels, confidence, availability = _contacts()
    return build_robot_motion_arrays(
        robot_id="g1",
        qpos=_qpos(),
        seconds=np.arange(4, dtype=np.float64) / 30.0,
        fps=30.0,
        contact_labels=labels,
        contact_confidence=confidence,
        contact_availability=availability,
        source_motion_sha256=SOURCE_SHA256,
        contact_source="kimodo_toe_contacts_6ch",
        hand_contact_source="unavailable_kimodo_default_false",
    )


def test_build_robot_motion_arrays_defines_a_pickle_free_versioned_schema() -> None:
    arrays = _build_payload()

    assert int(arrays["schema_version"]) == ROBOT_MOTION_SCHEMA_VERSION
    assert str(arrays["format"]) == ROBOT_MOTION_FORMAT
    assert str(arrays["robot_id"]) == "g1"
    assert str(arrays["qpos_layout"]) == QPOS_LAYOUT
    np.testing.assert_array_equal(arrays["root_qpos_names"], ROOT_QPOS_NAMES)
    np.testing.assert_array_equal(arrays["contact_label_names"], CONTACT_LABEL_NAMES)
    assert arrays["qpos"].shape == (4, 36)
    assert arrays["qpos"].dtype == np.dtype(np.float64)
    assert arrays["joint_names"].shape == (29,)
    np.testing.assert_array_equal(
        arrays["left_contact_segments"],
        np.asarray([[0, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["right_contact_segments"],
        np.asarray([[1, 3]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["flight_labels"],
        np.asarray([False, False, False, True]),
    )
    assert all(not np.asarray(value).dtype.hasobject for value in arrays.values())


def test_oli_export_expands_passive_ball_joint_quaternions() -> None:
    labels, confidence, availability = _contacts()
    arrays = build_robot_motion_arrays(
        robot_id="oli",
        qpos=_qpos(qpos_dim=68),
        seconds=np.arange(4, dtype=np.float64) / 30.0,
        fps=30.0,
        contact_labels=labels,
        contact_confidence=confidence,
        contact_availability=availability,
        source_motion_sha256=SOURCE_SHA256,
    )

    names = tuple(str(name) for name in arrays["joint_names"])
    assert len(names) == 61
    assert names[0] == "left_hip_pitch_joint"
    assert names[7:11] == (
        "left_A_achilles_rod_joint_quat_w",
        "left_A_achilles_rod_joint_quat_x",
        "left_A_achilles_rod_joint_quat_y",
        "left_A_achilles_rod_joint_quat_z",
    )


def test_write_robot_motion_npz_atomically_publishes_allow_pickle_false_archive(
    tmp_path: Path,
) -> None:
    labels, confidence, availability = _contacts()
    destination = tmp_path / "nested" / "robot_motion.npz"

    artifact = write_robot_motion_npz(
        destination,
        robot_id="G1",
        qpos=_qpos(),
        seconds=np.arange(4, dtype=np.float64) / 30.0,
        fps=30.0,
        contact_labels=labels,
        contact_confidence=confidence,
        contact_availability=availability,
        source_motion_sha256=SOURCE_SHA256,
    )

    assert artifact.path == destination.resolve()
    assert artifact.robot_id == "g1"
    assert artifact.frame_count == 4
    assert artifact.qpos_dim == 36
    assert artifact.fps == 30.0
    assert len(artifact.sha256) == 64
    assert destination.is_file()
    assert destination.stat().st_mode & 0o777 == 0o644
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp.npz"))
    with np.load(destination, allow_pickle=False) as archive:
        assert tuple(archive.files) == tuple(_build_payload())
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        np.testing.assert_array_equal(archive["qpos"], _qpos())


def test_write_robot_motion_npz_refuses_overwrite_by_default_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    labels, confidence, availability = _contacts()
    destination = tmp_path / "robot_motion.npz"
    common = {
        "robot_id": "g1",
        "qpos": _qpos(),
        "seconds": np.arange(4, dtype=np.float64) / 30.0,
        "fps": 30.0,
        "contact_labels": labels,
        "contact_confidence": confidence,
        "contact_availability": availability,
        "source_motion_sha256": SOURCE_SHA256,
    }
    first = write_robot_motion_npz(destination, **common)

    with pytest.raises(ArtifactError, match="already exists"):
        write_robot_motion_npz(destination, **common)

    changed = _qpos()
    changed[:, 0] = 0.25
    second = write_robot_motion_npz(
        destination,
        **{**common, "qpos": changed},
        overwrite=True,
    )

    assert second.sha256 != first.sha256
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["qpos"], changed)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"qpos": np.zeros((4, 35))}, "shape"),
        ({"qpos": np.full((4, 36), np.nan)}, "finite"),
        ({"seconds": np.asarray([0.0, 0.1, 0.2, 0.4])}, "uniformly sampled"),
        ({"fps": 0.0}, "positive"),
        ({"qpos": np.zeros((4, 36))}, "quaternions"),
        ({"contact_labels": np.zeros((4, 3))}, "labels must have shape"),
        ({"contact_confidence": np.full((4, 4), 2.0)}, r"\[0, 1\]"),
        ({"contact_availability": np.ones(3)}, "availability must have shape"),
        ({"source_motion_sha256": "bad"}, "64-character"),
        ({"contact_source": ""}, "must not be empty"),
    ),
)
def test_build_robot_motion_arrays_rejects_invalid_production_payloads(
    changes: dict[str, object],
    message: str,
) -> None:
    labels, confidence, availability = _contacts()
    arguments: dict[str, object] = {
        "robot_id": "g1",
        "qpos": _qpos(),
        "seconds": np.arange(4, dtype=np.float64) / 30.0,
        "fps": 30.0,
        "contact_labels": labels,
        "contact_confidence": confidence,
        "contact_availability": availability,
        "source_motion_sha256": SOURCE_SHA256,
    }
    arguments.update(changes)

    with pytest.raises(MotionValidationError, match=message):
        build_robot_motion_arrays(**arguments)  # type: ignore[arg-type]


def test_export_rejects_non_npz_path_before_creating_output(tmp_path: Path) -> None:
    labels, confidence, availability = _contacts()

    with pytest.raises(ArtifactError, match=r"\.npz"):
        write_robot_motion_npz(
            tmp_path / "robot_motion.pkl",
            robot_id="g1",
            qpos=_qpos(),
            seconds=np.arange(4, dtype=np.float64) / 30.0,
            fps=30.0,
            contact_labels=labels,
            contact_confidence=confidence,
            contact_availability=availability,
            source_motion_sha256=SOURCE_SHA256,
        )
    assert not (tmp_path / "robot_motion.pkl").exists()
