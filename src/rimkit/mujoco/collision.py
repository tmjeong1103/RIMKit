"""Deterministic MuJoCo collision candidates and signed-distance queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from rimkit.mujoco.model import MujocoModel
from rimkit.native import (
    BackendPreference,
    BackendSelection,
    mujoco_addresses,
    resolve_backend,
)
from rimkit.robots.profiles.schema import InitialCollisionProfile

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


def _readonly_float(value: Any, shape: tuple[int, ...]) -> FloatArray:
    result = np.asarray(value, dtype=np.float64).reshape(shape).copy(order="C")
    result.setflags(write=False)
    return result


def _readonly_int(value: Any, shape: tuple[int, ...]) -> IntArray:
    result = np.asarray(value, dtype=np.int32).reshape(shape).copy(order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CollisionCandidateSet:
    """Model-derived candidate pairs and the arm bodies/joints they may move."""

    geom_pairs: IntArray
    movable_body_ids: frozenset[int]
    arm_joint_names: tuple[str, ...]
    root_geom_count: int
    collision_geom_count: int
    raw_pair_count: int

    def __post_init__(self) -> None:
        pairs = np.asarray(self.geom_pairs, dtype=np.int32)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("collision geom pairs must have shape (N, 2)")
        object.__setattr__(self, "geom_pairs", _readonly_int(pairs, pairs.shape))
        object.__setattr__(
            self,
            "movable_body_ids",
            frozenset(int(body_id) for body_id in self.movable_body_ids),
        )
        object.__setattr__(
            self,
            "arm_joint_names",
            tuple(str(name) for name in self.arm_joint_names),
        )

    @property
    def pair_count(self) -> int:
        return int(len(self.geom_pairs))


@dataclass(frozen=True, slots=True)
class SignedDistanceBatch:
    """Closest signed geom distances at one already-forwarded model state."""

    distance: FloatArray
    fromto: FloatArray
    normal: FloatArray
    geom_pairs: IntArray
    body_pairs: IntArray
    source: IntArray
    contact_index: IntArray

    def __post_init__(self) -> None:
        count = int(np.asarray(self.distance).size)
        object.__setattr__(self, "distance", _readonly_float(self.distance, (count,)))
        object.__setattr__(self, "fromto", _readonly_float(self.fromto, (count, 6)))
        object.__setattr__(self, "normal", _readonly_float(self.normal, (count, 3)))
        object.__setattr__(self, "geom_pairs", _readonly_int(self.geom_pairs, (count, 2)))
        object.__setattr__(self, "body_pairs", _readonly_int(self.body_pairs, (count, 2)))
        object.__setattr__(self, "source", _readonly_int(self.source, (count,)))
        object.__setattr__(
            self,
            "contact_index",
            _readonly_int(self.contact_index, (count,)),
        )

    def __len__(self) -> int:
        return int(len(self.distance))


@dataclass(slots=True)
class _DistanceRecord:
    distance: float
    fromto: FloatArray
    normal: FloatArray
    geom1: int
    geom2: int
    body1: int
    body2: int
    source: int
    contact_index: int


def _matches_token(name: str, tokens: tuple[str, ...]) -> bool:
    lower_name = str(name).lower()
    return any(str(token).lower() in lower_name for token in tokens)


def _is_movable_body(name: str, profile: InitialCollisionProfile) -> bool:
    lower_name = str(name).lower()
    has_prefix = not profile.movable_body_prefixes or any(
        lower_name.startswith(prefix.lower()) for prefix in profile.movable_body_prefixes
    )
    return has_prefix and _matches_token(lower_name, profile.movable_body_tokens)


def _is_ground_body(name: str) -> bool:
    lower_name = str(name).lower()
    return lower_name == "world" or "floor" in lower_name or "ground" in lower_name


def _is_in_subtree(model: Any, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while True:
        if current == root_body_id:
            return True
        if current == 0:
            return False
        current = int(model.body_parentid[current])


def _ancestor_distance_within(
    model: Any,
    body1: int,
    body2: int,
    maximum_depth: int,
) -> bool:
    if body1 == body2:
        return True
    current = int(body1)
    for _ in range(maximum_depth):
        current = int(model.body_parentid[current])
        if current == body2:
            return True
        if current == 0:
            break
    current = int(body2)
    for _ in range(maximum_depth):
        current = int(model.body_parentid[current])
        if current == body1:
            return True
        if current == 0:
            break
    return False


def build_collision_candidates(
    adapter: MujocoModel,
    profile: InitialCollisionProfile,
) -> CollisionCandidateSet:
    """Build collision candidates from the configured root subtree."""

    model = adapter.model
    adapter.reset()
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.root_body_name))
    if root_body_id < 0:
        raise ValueError(f"MuJoCo model has no root body {profile.root_body_name!r}")

    collision_geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if not (int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0)
    )
    collision_geom_set = set(collision_geom_ids)
    root_geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if geom_id in collision_geom_set
        and _is_in_subtree(model, int(model.geom_bodyid[geom_id]), root_body_id)
    )

    exclude_signatures = {
        int(signature)
        for signature in np.asarray(
            getattr(model, "exclude_signature", np.empty(0, dtype=np.int32))
        ).reshape(-1)
    }
    raw_pair_keys: set[tuple[int, int]] = set()
    for geom1 in root_geom_ids:
        body1 = int(model.geom_bodyid[geom1])
        for geom2 in collision_geom_ids:
            if geom1 == geom2:
                continue
            key = min(geom1, geom2), max(geom1, geom2)
            if key in raw_pair_keys:
                continue
            body2 = int(model.geom_bodyid[geom2])
            if body1 == body2:
                continue
            signature12 = ((body1 + 1) << 16) + body2 + 1
            signature21 = ((body2 + 1) << 16) + body1 + 1
            if signature12 in exclude_signatures or signature21 in exclude_signatures:
                continue
            parent1 = int(model.body_parentid[body1])
            parent2 = int(model.body_parentid[body2])
            if (parent1 == body2 and body2 != 0) or (parent2 == body1 and body1 != 0):
                continue
            can_collide = (int(model.geom_contype[geom1]) & int(model.geom_conaffinity[geom2])) or (
                int(model.geom_contype[geom2]) & int(model.geom_conaffinity[geom1])
            )
            if not can_collide:
                continue
            raw_pair_keys.add(key)

    movable_body_ids = frozenset(
        body_id
        for body_id, body_name in enumerate(adapter.body_names)
        if _is_movable_body(body_name, profile)
    )
    filtered_pairs: list[tuple[int, int]] = []
    for geom1, geom2 in sorted(raw_pair_keys):
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if _is_ground_body(adapter.body_names[body1]) or _is_ground_body(adapter.body_names[body2]):
            continue
        if _ancestor_distance_within(
            model,
            body1,
            body2,
            profile.ancestor_skip_depth,
        ):
            continue
        if body1 not in movable_body_ids and body2 not in movable_body_ids:
            continue
        filtered_pairs.append((geom1, geom2))

    arm_joint_names = tuple(
        name
        for name in adapter.rev_joint_names
        if _matches_token(name, profile.movable_joint_tokens)
    )
    pairs = np.asarray(filtered_pairs, dtype=np.int32).reshape(-1, 2)
    return CollisionCandidateSet(
        geom_pairs=pairs,
        movable_body_ids=movable_body_ids,
        arm_joint_names=arm_joint_names,
        root_geom_count=len(root_geom_ids),
        collision_geom_count=len(collision_geom_ids),
        raw_pair_count=len(raw_pair_keys),
    )


def _pair_key(geom1: int, geom2: int) -> tuple[int, int]:
    return min(geom1, geom2), max(geom1, geom2)


def _truncate(records: list[_DistanceRecord], limit: int | None) -> None:
    records.sort(key=lambda record: record.distance)
    if limit is not None and len(records) > limit:
        del records[limit:]


def query_signed_distances(
    adapter: MujocoModel,
    geom_pairs: IntArray,
    *,
    limit: int | None = None,
    distance_limit: float = np.inf,
    sanity_check: bool = True,
    lower_bound_tolerance: float = 1e-6,
    fromto_distance_tolerance: float = 1e-4,
    backend: BackendPreference | BackendSelection = "python",
) -> SignedDistanceBatch:
    """Query candidates without forwarding in deterministic native order."""

    backend_selection = resolve_backend(backend)
    pairs = np.ascontiguousarray(geom_pairs, dtype=np.int32).reshape(-1, 2)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0 or len(pairs) == 0:
        return SignedDistanceBatch(
            distance=np.empty(0, dtype=np.float64),
            fromto=np.empty((0, 6), dtype=np.float64),
            normal=np.empty((0, 3), dtype=np.float64),
            geom_pairs=np.empty((0, 2), dtype=np.int32),
            body_pairs=np.empty((0, 2), dtype=np.int32),
            source=np.empty(0, dtype=np.int32),
            contact_index=np.empty(0, dtype=np.int32),
        )
    if not np.isfinite(distance_limit) and not np.isposinf(distance_limit):
        raise ValueError("distance_limit must be finite or positive infinity")

    model = adapter.model
    data = adapter.data
    if np.any(pairs < 0) or np.any(pairs >= int(model.ngeom)):
        raise ValueError("geom_pairs contains an out-of-range MuJoCo geom ID")
    if backend_selection.is_native:
        model_address, data_address = mujoco_addresses(model, data)
        arrays = backend_selection.module.signed_distance_arrays(
            model_address,
            data_address,
            pairs,
            float(distance_limit),
            -1 if limit is None else int(limit),
            True,
            bool(sanity_check),
            float(lower_bound_tolerance),
            float(fromto_distance_tolerance),
        )
        count = int(arrays["n_result"])
        native_distance = np.asarray(arrays["dist"], dtype=np.float64)
        if native_distance.shape != (count,):
            raise RuntimeError("Native signed-distance result count is inconsistent.")
        return SignedDistanceBatch(
            distance=native_distance,
            fromto=np.asarray(arrays["fromto"], dtype=np.float64),
            normal=np.asarray(arrays["normal"], dtype=np.float64),
            geom_pairs=np.asarray(arrays["geom_pair"], dtype=np.int32),
            body_pairs=np.asarray(arrays["body_pair"], dtype=np.int32),
            source=np.asarray(arrays["source"], dtype=np.int32),
            contact_index=np.asarray(arrays["contact_idx"], dtype=np.int32),
        )

    candidate_keys = {_pair_key(int(pair[0]), int(pair[1])) for pair in pairs}
    contact_keys: set[tuple[int, int]] = set()
    records: list[_DistanceRecord] = []

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 < 0 or geom2 < 0:
            continue
        key = _pair_key(geom1, geom2)
        if key not in candidate_keys or key in contact_keys:
            continue
        distance = float(contact.dist)
        normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
        position = np.asarray(contact.pos, dtype=np.float64).copy()
        fromto = np.concatenate(
            (
                position - 0.5 * distance * normal,
                position + 0.5 * distance * normal,
            )
        )
        records.append(
            _DistanceRecord(
                distance=distance,
                fromto=fromto,
                normal=normal,
                geom1=geom1,
                geom2=geom2,
                body1=int(model.geom_bodyid[geom1]),
                body2=int(model.geom_bodyid[geom2]),
                source=1,
                contact_index=contact_index,
            )
        )
        contact_keys.add(key)

    current_limit = float(distance_limit)
    if limit is not None and len(records) >= limit:
        _truncate(records, limit)
        current_limit = min(current_limit, records[-1].distance)

    lower_bounds = np.empty(len(pairs), dtype=np.float64)
    for pair_index, pair in enumerate(pairs):
        geom1 = int(pair[0])
        geom2 = int(pair[1])
        delta = np.asarray(data.geom_xpos[geom2]) - np.asarray(data.geom_xpos[geom1])
        lower_bounds[pair_index] = (
            float(np.linalg.norm(delta))
            - float(model.geom_rbound[geom1])
            - float(model.geom_rbound[geom2])
        )
    pair_order = (
        sorted(range(len(pairs)), key=lambda index: lower_bounds[index])
        if limit is not None
        else range(len(pairs))
    )

    fromto = np.zeros(6, dtype=np.float64)
    for pair_index in pair_order:
        if limit is not None and len(records) >= limit and lower_bounds[pair_index] > current_limit:
            break
        geom1 = int(pairs[pair_index, 0])
        geom2 = int(pairs[pair_index, 1])
        if _pair_key(geom1, geom2) in contact_keys:
            continue
        fromto[:] = 0.0
        query_limit = current_limit if limit is not None else float(distance_limit)
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                geom1,
                geom2,
                query_limit,
                fromto,
            )
        )
        if sanity_check:
            if distance < lower_bounds[pair_index] - lower_bound_tolerance:
                continue
            if not np.isfinite(distance) or not np.isfinite(fromto).all():
                continue
            fromto_distance = float(np.linalg.norm(fromto[3:] - fromto[:3]))
            if distance <= 1e-12 and fromto_distance <= 1e-12:
                continue
            if abs(fromto_distance - distance) > fromto_distance_tolerance:
                continue

        records.append(
            _DistanceRecord(
                distance=distance,
                fromto=fromto.copy(),
                normal=np.zeros(3, dtype=np.float64),
                geom1=geom1,
                geom2=geom2,
                body1=int(model.geom_bodyid[geom1]),
                body2=int(model.geom_bodyid[geom2]),
                source=0,
                contact_index=-1,
            )
        )
        if limit is not None:
            _truncate(records, limit)
            if len(records) >= limit:
                current_limit = min(float(distance_limit), records[-1].distance)

    _truncate(records, limit)
    count = len(records)
    return SignedDistanceBatch(
        distance=np.asarray([record.distance for record in records], dtype=np.float64),
        fromto=np.asarray([record.fromto for record in records], dtype=np.float64).reshape(
            count, 6
        ),
        normal=np.asarray([record.normal for record in records], dtype=np.float64).reshape(
            count, 3
        ),
        geom_pairs=np.asarray(
            [(record.geom1, record.geom2) for record in records], dtype=np.int32
        ).reshape(count, 2),
        body_pairs=np.asarray(
            [(record.body1, record.body2) for record in records], dtype=np.int32
        ).reshape(count, 2),
        source=np.asarray([record.source for record in records], dtype=np.int32),
        contact_index=np.asarray([record.contact_index for record in records], dtype=np.int32),
    )


__all__ = [
    "CollisionCandidateSet",
    "SignedDistanceBatch",
    "build_collision_candidates",
    "query_signed_distances",
]
