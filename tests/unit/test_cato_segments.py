"""Production export shares the renderer's existing macro implementation."""

import json

import numpy as np
import pytest

from rimkit.exceptions import MotionValidationError
from rimkit.export.motion import write_robot_motion_npz
from rimkit.motion.cato_segments import build_cato_macro_segments, lower_body_qpos_indices
from rimkit.render.contact import PreviewContactState, _macro_segments
from rimkit.robots.registry import get_robot


def test_preview_macro_partition_and_labels_survive_production_export(tmp_path):
    assert _macro_segments is build_cato_macro_segments
    n, fps = 80, 30.0
    qpos = np.zeros((n, get_robot("g1").expected_nq))
    qpos[:, 3] = 1
    qpos[30:60, 0] = np.linspace(0, 1, 30)
    qpos[60:, 0] = 1
    labels = np.zeros((n, 4), dtype=bool)
    labels[:40, 0] = True
    labels[35:, 1] = True
    ranges, boundaries = build_cato_macro_segments(qpos, labels, fps, lower_body_qpos_indices("g1"))
    state = PreviewContactState(
        fps,
        np.arange(n) / fps,
        labels,
        labels.astype(float),
        np.array([True, True, False, False]),
        ~(labels[:, 0] | labels[:, 1]),
        ranges,
        boundaries,
        "synthetic-test",
        "unavailable-test",
    )
    path = tmp_path / "robot_motion.npz"
    write_robot_motion_npz(
        path,
        robot_id="g1",
        qpos=qpos,
        seconds=state.seconds,
        fps=fps,
        contact_labels=state.labels,
        contact_confidence=state.confidence,
        contact_availability=state.availability,
        flight_labels=state.flight,
        cato_segment_frames=state.segment_ranges,
        cato_segments=state.segment_boundaries,
        source_motion_sha256="a" * 64,
        contact_source=state.contact_source,
    )
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["cato_segments"], state.segment_boundaries)
        np.testing.assert_array_equal(data["cato_segment_frames"], state.segment_ranges)
        np.testing.assert_array_equal(data["contact_labels"], state.labels)
        provenance = json.loads(str(data["cato_segments_provenance"].item()))
        assert provenance["config"]["min_segment_time"] == 0.4
        assert provenance["source_motion_sha256"] == "a" * 64
        assert str(data["contact_annotation_status"].item()) == "unreviewed"


def test_export_rejects_unpaired_macro_metadata(tmp_path):
    from rimkit.export.motion import build_robot_motion_arrays

    qpos = np.zeros((2, 36))
    qpos[:, 3] = 1
    with pytest.raises(MotionValidationError, match="supplied together"):
        build_robot_motion_arrays(
            robot_id="g1",
            qpos=qpos,
            seconds=[0, 1 / 30],
            fps=30,
            contact_labels=np.zeros((2, 4), dtype=bool),
            contact_confidence=np.zeros((2, 4)),
            contact_availability=[1, 1, 0, 0],
            source_motion_sha256="a" * 64,
            cato_segments=[0, 1],
        )
