from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rimkit.robots.profiles import (
    ARA_PROFILES,
    ROBOT_NEUTRAL_ARA_PROFILE,
    AraProfile,
    get_ara_profile,
)


def test_all_supported_robots_share_one_immutable_ara_profile() -> None:
    assert tuple(ARA_PROFILES) == (
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
    for robot_id in ARA_PROFILES:
        assert get_ara_profile(robot_id) is ROBOT_NEUTRAL_ARA_PROFILE

    with pytest.raises(TypeError):
        ARA_PROFILES["g1"] = AraProfile()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        ROBOT_NEUTRAL_ARA_PROFILE.slip_weight = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    (
        "slip_weight",
        "floor_lock_weight",
        "scale_regularization_weight",
        "shift_regularization_weight",
        "scale_xy_bound",
        "shift_xy_bound",
    ),
)
def test_ara_profile_rejects_negative_or_nonfinite_tuning(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        AraProfile(**{field_name: -1.0})
    with pytest.raises(ValueError, match=field_name):
        AraProfile(**{field_name: float("nan")})


@pytest.mark.parametrize("field_name", ("ground_target_z", "toe_floor_offset"))
def test_ara_profile_rejects_nonfinite_floor_values(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        AraProfile(**{field_name: float("inf")})
