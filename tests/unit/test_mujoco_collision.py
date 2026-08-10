from __future__ import annotations

import numpy as np
import pytest

from core_retarget.mujoco import (
    MujocoModel,
    build_collision_candidates,
    query_signed_distances,
)
from core_retarget.robots.profiles import get_initial_collision_profile

EXPECTED_CANDIDATES = {
    # root geoms, collision geoms, raw pairs, filtered pairs, movable bodies, arm joints
    "g1": (39, 40, 720, 444, 14, 14),
    "h1": (20, 21, 191, 108, 12, 8),
    "h2": (33, 34, 507, 261, 16, 14),
    "r1": (24, 25, 268, 111, 22, 14),
    "k1": (15, 16, 114, 63, 12, 10),
}


@pytest.mark.mujoco
@pytest.mark.parametrize(
    ("robot_id", "expected"),
    tuple(EXPECTED_CANDIDATES.items()),
)
def test_collision_candidates_match_frozen_stage3_counts(
    robot_id: str,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    model = MujocoModel.from_robot(robot_id)
    candidates = build_collision_candidates(
        model,
        get_initial_collision_profile(robot_id),
    )
    root_count, collision_count, raw_count, pair_count, body_count, joint_count = expected

    assert candidates.root_geom_count == root_count
    assert candidates.collision_geom_count == collision_count
    assert candidates.raw_pair_count == raw_count
    assert candidates.pair_count == pair_count
    assert len(candidates.movable_body_ids) == body_count
    assert len(candidates.arm_joint_names) == joint_count
    assert candidates.geom_pairs.shape == (pair_count, 2)
    assert candidates.geom_pairs.dtype == np.dtype(np.int32)
    assert not candidates.geom_pairs.flags.writeable
    assert np.all(candidates.geom_pairs[:, 0] < candidates.geom_pairs[:, 1])
    assert candidates.geom_pairs.tolist() == sorted(candidates.geom_pairs.tolist())
    assert len({tuple(pair) for pair in candidates.geom_pairs.tolist()}) == pair_count
    with pytest.raises(ValueError):
        candidates.geom_pairs[0, 0] = -1


@pytest.mark.mujoco
def test_g1_rubber_hand_meshes_participate_in_self_collision() -> None:
    model = MujocoModel.from_robot("g1")
    candidates = build_collision_candidates(
        model,
        get_initial_collision_profile("g1"),
    )

    rubber_hand_mesh_ids = {
        int(model.model.mesh(name).id) for name in ("left_rubber_hand", "right_rubber_hand")
    }
    hand_geom_ids: set[int] = set()
    for geom_id in range(int(model.model.ngeom)):
        mesh_id = int(model.model.geom_dataid[geom_id])
        if mesh_id not in rubber_hand_mesh_ids:
            continue
        hand_geom_ids.add(geom_id)
        assert int(model.model.geom_contype[geom_id]) == 1
        assert int(model.model.geom_conaffinity[geom_id]) == 1

    assert len(hand_geom_ids) == 2
    candidate_geom_ids = set(candidates.geom_pairs.reshape(-1).tolist())
    assert hand_geom_ids <= candidate_geom_ids


@pytest.mark.mujoco
@pytest.mark.parametrize("robot_id", tuple(EXPECTED_CANDIDATES))
def test_signed_distance_query_has_read_only_reference_contract(robot_id: str) -> None:
    model = MujocoModel.from_robot(robot_id)
    candidates = build_collision_candidates(
        model,
        get_initial_collision_profile(robot_id),
    )
    model.reset()

    batch = query_signed_distances(model, candidates.geom_pairs, limit=32)
    count = len(batch)

    assert 0 < count <= 32
    assert batch.distance.shape == (count,)
    assert batch.fromto.shape == (count, 6)
    assert batch.normal.shape == (count, 3)
    assert batch.geom_pairs.shape == (count, 2)
    assert batch.body_pairs.shape == (count, 2)
    assert batch.source.shape == (count,)
    assert batch.contact_index.shape == (count,)
    assert batch.distance.dtype == np.dtype(np.float64)
    assert batch.fromto.dtype == np.dtype(np.float64)
    assert batch.normal.dtype == np.dtype(np.float64)
    assert batch.geom_pairs.dtype == np.dtype(np.int32)
    assert batch.body_pairs.dtype == np.dtype(np.int32)
    assert batch.source.dtype == np.dtype(np.int32)
    assert batch.contact_index.dtype == np.dtype(np.int32)

    for array in (
        batch.distance,
        batch.fromto,
        batch.normal,
        batch.geom_pairs,
        batch.body_pairs,
        batch.source,
        batch.contact_index,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        batch.distance[0] = 0.0

    assert np.isfinite(batch.distance).all()
    assert np.isfinite(batch.fromto).all()
    assert np.isfinite(batch.normal).all()
    assert np.all(batch.distance[:-1] <= batch.distance[1:])
    assert set(batch.source.tolist()) <= {0, 1}

    candidate_keys = {tuple(pair) for pair in candidates.geom_pairs.tolist()}
    for index, (geom1_value, geom2_value) in enumerate(batch.geom_pairs):
        geom1 = int(geom1_value)
        geom2 = int(geom2_value)
        assert (min(geom1, geom2), max(geom1, geom2)) in candidate_keys
        assert int(batch.body_pairs[index, 0]) == int(model.model.geom_bodyid[geom1])
        assert int(batch.body_pairs[index, 1]) == int(model.model.geom_bodyid[geom2])

    closest_point_distance = np.linalg.norm(
        batch.fromto[:, 3:] - batch.fromto[:, :3],
        axis=1,
    )
    contact = batch.source == 1
    noncontact = ~contact
    np.testing.assert_allclose(
        closest_point_distance[contact],
        np.abs(batch.distance[contact]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        closest_point_distance[noncontact],
        batch.distance[noncontact],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        batch.normal[noncontact],
        np.zeros((int(np.count_nonzero(noncontact)), 3), dtype=np.float64),
    )
    assert np.all(batch.contact_index[contact] >= 0)
    assert np.all(batch.contact_index[noncontact] == -1)


@pytest.mark.mujoco
def test_signed_distance_query_empty_and_limit_validation() -> None:
    model = MujocoModel.from_robot("k1")
    empty_pairs = np.empty((0, 2), dtype=np.int32)
    candidates = build_collision_candidates(
        model,
        get_initial_collision_profile("k1"),
    )

    empty = query_signed_distances(model, empty_pairs, limit=32)

    assert len(empty) == 0
    assert empty.fromto.shape == (0, 6)
    assert empty.normal.shape == (0, 3)
    assert empty.geom_pairs.shape == (0, 2)
    assert empty.body_pairs.shape == (0, 2)
    assert not empty.distance.flags.writeable
    with pytest.raises(ValueError, match="limit"):
        query_signed_distances(model, empty_pairs, limit=-1)
    with pytest.raises(ValueError, match="distance_limit"):
        query_signed_distances(model, candidates.geom_pairs[:1], distance_limit=np.nan)
