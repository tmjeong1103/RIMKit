from __future__ import annotations

import numpy as np
import pytest

from core_retarget.mujoco import MujocoModel
from core_retarget.mujoco.robot_kinematics import derive_neutral_geometry
from core_retarget.robots.profiles.k1 import K1_DMR_PROFILE

K1_NEUTRAL_LINK_LENGTHS = {
    "base_spine": 0.11128791488746655,
    "spine_neck": 0.31901920004915074,
    "neck_rs": 0.14600000000000002,
    "rs_re": 0.18482694608741437,
    "re_rw": 0.21869999999999998,
    "neck_ls": 0.14600000000000002,
    "ls_le": 0.18482694608741437,
    "le_lw": 0.21869999999999998,
    "base_rp": 0.125,
    "rp_rk": 0.32709784468871084,
    "rk_ra": 0.294,
    "ra_rt": 0.1341640786499874,
    "base_lp": 0.125,
    "lp_lk": 0.32709784468871084,
    "lk_la": 0.294,
    "la_lt": 0.1341640786499874,
}


@pytest.mark.mujoco
def test_k1_neutral_link_lengths_match_legacy_audit() -> None:
    robot = MujocoModel.from_robot("k1")

    geometry = derive_neutral_geometry(
        robot,
        K1_DMR_PROFILE.joi_bodies,
        link_length_base_reference=K1_DMR_PROFILE.link_length_base_reference,
    )

    assert tuple(geometry.link_lengths) == tuple(K1_NEUTRAL_LINK_LENGTHS)
    np.testing.assert_allclose(
        tuple(geometry.link_lengths.values()),
        tuple(K1_NEUTRAL_LINK_LENGTHS.values()),
        rtol=0.0,
        atol=1e-12,
    )
