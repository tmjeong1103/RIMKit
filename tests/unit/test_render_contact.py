from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core_retarget.exceptions import MotionValidationError
from core_retarget.motion.soma import load_soma_motion
from core_retarget.render.contact import PreviewContactState, build_preview_contact_state
from core_retarget.render.legacy_visualization import annotate_frame, load_overlay_runtime
from core_retarget.robots.registry import get_robot

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MOTION_ROOT = REPOSITORY_ROOT / "examples/motions/kimodo/soma_rp_v11"
RUN_ROOT = REPOSITORY_ROOT / "runs/kimodo-stage3"


def _state(*, confidence: np.ndarray | None = None) -> PreviewContactState:
    labels = np.asarray(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, False, False],
        ]
    )
    return PreviewContactState(
        fps=30.0,
        seconds=np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0]),
        labels=labels,
        confidence=labels.astype(np.float64) if confidence is None else confidence,
        availability=np.asarray([True, True, False, False]),
        flight=np.asarray([False, False, False]),
        segment_ranges=np.asarray([[0, 2], [2, 3]]),
        segment_boundaries=np.asarray([0.0, 2.0 / 3.0, 1.0]),
        contact_source="test-source",
        hand_contact_source="unavailable-test",
    )


def test_preview_contact_state_defensively_copies_object_free_arrays() -> None:
    labels = np.asarray(
        [[True, False, False, False], [False, True, False, False]],
        dtype=bool,
    )
    state = PreviewContactState(
        fps=30.0,
        seconds=np.asarray([0.0, 1.0 / 30.0]),
        labels=labels,
        confidence=labels.astype(np.float64),
        availability=np.asarray([True, True, False, False]),
        flight=np.asarray([False, False]),
        segment_ranges=np.asarray([[0, 2]]),
        segment_boundaries=np.asarray([0.0, 1.0]),
        contact_source="test-source",
        hand_contact_source="unavailable-test",
    )
    labels[:] = False

    assert state.labels.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
    ]
    assert not state.labels.flags.writeable
    assert not state.confidence.flags.writeable
    arrays = state.reference_arrays()
    assert arrays["contact_label_names"].tolist() == [
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
    ]
    assert all(not np.asarray(value).dtype.hasobject for value in arrays.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"labels": np.zeros((3, 3), dtype=bool)}, "labels"),
        ({"confidence": np.full((3, 4), 1.1)}, "confidence"),
        ({"availability": np.asarray([True, False, False, False])}, "Unavailable"),
        ({"segment_ranges": np.asarray([[1, 3]])}, "partition"),
    ),
)
def test_preview_contact_state_rejects_invalid_contracts(
    kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    valid = {
        "fps": 30.0,
        "seconds": np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0]),
        "labels": np.asarray(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, False, False],
            ]
        ),
        "confidence": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        "availability": np.asarray([True, True, False, False]),
        "flight": np.asarray([False, False, False]),
        "segment_ranges": np.asarray([[0, 3]]),
        "segment_boundaries": np.asarray([0.0, 1.0]),
        "contact_source": "test",
        "hand_contact_source": "unavailable-test",
    }
    valid.update(kwargs)

    with pytest.raises(MotionValidationError, match=message):
        PreviewContactState(**valid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("motion_id", "expected_counts", "expected_flight"),
    (
        ("stand_walk_run_stop", [90, 88, 0, 0], 16),
        ("march_in_place_contacts", [187, 170, 0, 0], 0),
    ),
)
def test_bundled_contact_state_matches_verified_schedule(
    motion_id: str,
    expected_counts: list[int],
    expected_flight: int,
) -> None:
    motion = load_soma_motion(MOTION_ROOT / f"{motion_id}.npz")
    qpos_path = RUN_ROOT / motion_id / "h2/stages/3_initial_collision.npz"
    if qpos_path.is_file():
        with np.load(qpos_path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos_cc_smt_array"])
    else:
        qpos = np.zeros((motion.frame_count, get_robot("h2").expected_nq))
        qpos[:, 3] = 1.0
    contacts = build_preview_contact_state(motion, qpos=qpos, robot_id="h2")

    assert contacts.labels.sum(axis=0).tolist() == expected_counts
    assert int(contacts.flight.sum()) == expected_flight
    assert contacts.availability.tolist() == [True, True, False, False]
    assert contacts.segment_ranges[0, 0] == 0
    assert contacts.segment_ranges[-1, 1] == motion.frame_count


def test_legacy_overlay_draws_panel_without_mutating_rgb() -> None:
    pytest.importorskip("PIL")
    rgb = np.full((720, 1280, 3), 127, dtype=np.uint8)
    original = rgb.copy()
    runtime = load_overlay_runtime(width=1280, height=720)

    annotated = annotate_frame(
        rgb,
        motion_name="example_motion",
        tick=1,
        contacts=_state(),
        runtime=runtime,
    )

    np.testing.assert_array_equal(rgb, original)
    assert annotated.shape == rgb.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated[16:305, 18:487], original[16:305, 18:487])
    np.testing.assert_array_equal(annotated[400:500, 800:900], original[400:500, 800:900])


def test_legacy_overlay_omits_contact_segment_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PIL")
    rgb = np.full((720, 1280, 3), 127, dtype=np.uint8)
    runtime = load_overlay_runtime(width=1280, height=720)
    draw_factory = runtime.image_draw.Draw
    rendered_text: list[str] = []

    class RecordingDraw:
        def __init__(self, draw: object) -> None:
            self._draw = draw

        def text(self, xy: object, text: str, **kwargs: object) -> None:
            rendered_text.append(text)
            self._draw.text(xy, text, **kwargs)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._draw, name)

    monkeypatch.setattr(
        runtime.image_draw,
        "Draw",
        lambda image, mode: RecordingDraw(draw_factory(image, mode)),
    )

    annotate_frame(
        rgb,
        motion_name="example_motion",
        tick=1,
        contacts=_state(),
        runtime=runtime,
    )

    assert "CONTACT STATE" in rendered_text
    assert all("contact segment" not in text.casefold() for text in rendered_text)
    assert all("phase" not in text.casefold() for text in rendered_text)
