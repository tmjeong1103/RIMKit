from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rimkit.motion import (
    build_contact_schedule,
    extract_soma_joi,
    load_soma_motion,
)

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "motions" / "kimodo" / "soma_rp_v11" / "stand_walk_run_stop.npz"


def test_contact_schedule_public_contract_is_immutable() -> None:
    motion = load_soma_motion(EXAMPLE)
    source_joi = extract_soma_joi(motion)

    schedule = build_contact_schedule(motion, source_joi=source_joi)

    assert schedule.frame_count == motion.frame_count
    assert schedule.fps == motion.fps
    assert schedule.contact_source == (
        "kimodo_toe_contacts_6ch+geometry_gap_fill+flight_aware_bridge"
    )
    assert schedule.left_confidence.shape == (motion.frame_count,)
    assert schedule.right_confidence.shape == (motion.frame_count,)
    assert np.all((schedule.left_confidence >= 0.0) & (schedule.left_confidence <= 1.0))
    assert np.all((schedule.right_confidence >= 0.0) & (schedule.right_confidence <= 1.0))

    for value in schedule.reference_arrays().values():
        if value.ndim > 0:
            assert not value.flags.writeable
    with pytest.raises(ValueError):
        schedule.left_confidence[0] = 0.5


def test_reference_aliases_share_the_public_confidence_arrays() -> None:
    schedule = build_contact_schedule(load_soma_motion(EXAMPLE))
    arrays = schedule.reference_arrays()

    assert arrays["left_contact_confidence"] is schedule.left_confidence
    assert arrays["l_contact_confidence"] is schedule.left_confidence
    assert arrays["right_contact_confidence"] is schedule.right_confidence
    assert arrays["r_contact_confidence"] is schedule.right_confidence
