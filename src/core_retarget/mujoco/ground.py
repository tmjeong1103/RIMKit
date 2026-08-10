"""Physical sole-to-ground distance helpers for foot placement adjustment."""

from __future__ import annotations

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_retarget.mujoco.model import MujocoModel

FloatArray = NDArray[np.float64]


def find_ground_geoms(adapter: MujocoModel) -> tuple[int, ...]:
    """Find the scene plane geoms used as ground by the research pipeline."""

    model = adapter.model
    result: list[int] = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        body_name = adapter.body_names[int(model.geom_bodyid[geom_id])].lower()
        geom_name_value = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_name = "" if geom_name_value is None else str(geom_name_value).lower()
        if body_name == "world" or "floor" in geom_name or "ground" in geom_name:
            result.append(geom_id)
    if not result:
        raise ValueError("No ground plane geom was found in the MuJoCo model")
    return tuple(result)


def find_body_collision_geoms(
    adapter: MujocoModel,
    body_name: str,
    *,
    include_descendants: bool = False,
) -> tuple[int, ...]:
    """Find collision-enabled geoms on one body or its subtree."""

    model = adapter.model
    try:
        body_id = adapter.body_names.index(body_name)
    except ValueError as exc:
        raise ValueError(f"Unknown MuJoCo body: {body_name}") from exc
    body_ids = {body_id}
    if include_descendants:
        for candidate in range(int(model.nbody)):
            parent = candidate
            while parent != 0 and parent != body_id:
                parent = int(model.body_parentid[parent])
            if parent == body_id:
                body_ids.add(candidate)
    result = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_ids
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    )
    if not result:
        raise ValueError(f"No collision geom was found on body: {body_name}")
    return result


def minimum_geom_signed_distance(
    adapter: MujocoModel,
    first_geoms: tuple[int, ...],
    second_geoms: tuple[int, ...],
) -> float:
    """Return the minimum MuJoCo signed distance between two geom groups."""

    fromto = np.zeros(6, dtype=np.float64)
    distances = [
        float(
            mujoco.mj_geomDistance(
                adapter.model,
                adapter.data,
                first,
                second,
                1.0,
                fromto,
            )
        )
        for first in first_geoms
        for second in second_geoms
    ]
    if not distances:
        raise ValueError("At least one geom is required in each group")
    return float(min(distances))


def foot_ground_signed_distance(
    adapter: MujocoModel,
    qpos: ArrayLike,
    *,
    foot_body_name: str,
) -> FloatArray:
    """Measure the physical sole mesh's signed ground distance per frame."""

    trajectory = np.asarray(qpos, dtype=np.float64)
    ground = find_ground_geoms(adapter)
    foot = find_body_collision_geoms(adapter, foot_body_name, include_descendants=True)
    output = np.empty(len(trajectory), dtype=np.float64)
    for tick, pose in enumerate(trajectory):
        adapter.forward(pose)
        output[tick] = minimum_geom_signed_distance(adapter, ground, foot)
    return output


def pose_dependent_toe_target_z(
    adapter: MujocoModel,
    qpos: ArrayLike,
    *,
    toe_body_name: str,
    foot_body_name: str,
    ground_clearance: float,
) -> tuple[FloatArray, FloatArray]:
    """Return toe height that places the current oriented sole on the plane."""

    trajectory = np.asarray(qpos, dtype=np.float64)
    ground = find_ground_geoms(adapter)
    foot = find_body_collision_geoms(adapter, foot_body_name, include_descendants=True)
    target_z = np.empty(len(trajectory), dtype=np.float64)
    distance = np.empty(len(trajectory), dtype=np.float64)
    for tick, pose in enumerate(trajectory):
        adapter.forward(pose)
        value = minimum_geom_signed_distance(adapter, ground, foot)
        toe_z = float(adapter.get_body_transform(toe_body_name)[2, 3])
        target_z[tick] = toe_z - value + float(ground_clearance)
        distance[tick] = value
    return target_z, distance


__all__ = [
    "find_body_collision_geoms",
    "find_ground_geoms",
    "foot_ground_signed_distance",
    "minimum_geom_signed_distance",
    "pose_dependent_toe_target_z",
]
