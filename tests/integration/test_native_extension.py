from __future__ import annotations

import mujoco
import numpy as np

from core_retarget.mujoco import (
    MujocoModel,
    build_collision_candidates,
    query_signed_distances,
)
from core_retarget.mujoco.ik import BodyPositionIKSolver
from core_retarget.native import NATIVE_API_VERSION, NATIVE_MUJOCO_VERSION, resolve_backend
from core_retarget.robots.profiles import get_initial_collision_profile

_MODEL_XML = """
<mujoco model="native-extension-test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.02"/>
      <body name="tip" pos="1 0 0"><geom type="sphere" size="0.03"/></body>
    </body>
  </worldbody>
</mujoco>
"""

_PLANAR_MODEL_XML = """
<mujoco model="planar-native-test">
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05"/>
      <body name="tip" pos="1 0 0">
        <geom type="sphere" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _TinyAdapter:
    rev_joint_names = ("hinge",)
    pri_joint_names: tuple[str, ...] = ()
    rev_pri_joint_names = ("hinge",)

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def copy_state_from(self, other: _TinyAdapter) -> None:
        self.data.qpos[:] = other.data.qpos
        mujoco.mj_forward(self.model, self.data)


def test_compiled_extension_metadata_and_frozen_symbols() -> None:
    backend = resolve_backend("native")

    assert backend.selected == "native"
    assert backend.native_info["api_version"] == NATIVE_API_VERSION
    assert backend.native_info["mujoco_version"] == NATIVE_MUJOCO_VERSION
    assert backend.native_info["mujoco_runtime_version"] == NATIVE_MUJOCO_VERSION
    assert callable(backend.module.signed_distance_arrays)
    assert callable(backend.module.solve_body_position_ik)
    assert callable(backend.module.solve_planar_base_ik)


def test_compiled_body_ik_matches_reference_ordered_python() -> None:
    backend = resolve_backend("native")
    python_adapter = _TinyAdapter()
    native_adapter = _TinyAdapter()
    parameters = {
        "max_iterations": 12,
        "revolute_step": 0.5,
        "revolute_update_limit": 0.1,
        "damping": 1e-4,
        "joint_limit_probe": 0.02,
        "reference_ordered": True,
    }
    python_solver = BodyPositionIKSolver(python_adapter, backend="python", **parameters)
    native_solver = BodyPositionIKSolver(native_adapter, backend=backend, **parameters)
    desired_angle = 0.35
    target = np.asarray([np.cos(desired_angle), np.sin(desired_angle), 0.0])
    for adapter, solver in (
        (python_adapter, python_solver),
        (native_adapter, native_solver),
    ):
        current = adapter.data.xpos[adapter.model.body("tip").id].copy()
        solver.add_target("tip", current, target)

    python_result = python_solver.solve()
    native_result = native_solver.solve()

    np.testing.assert_array_equal(native_result.qpos, python_result.qpos)
    np.testing.assert_array_equal(native_result.joint_qpos, python_result.joint_qpos)
    np.testing.assert_array_equal(native_result.errors, python_result.errors)
    assert native_result.error == python_result.error
    assert native_result.best_iteration == python_result.best_iteration


def test_compiled_planar_base_ik_matches_frozen_native_contract() -> None:
    backend = resolve_backend("native")
    model = mujoco.MjModel.from_xml_string(_PLANAR_MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    result = backend.module.solve_planar_base_ik(
        int(model._address),
        int(data._address),
        np.asarray([model.body("tip").id], dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
        np.asarray([[1.10, 0.30, 0.0]], dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.arange(6, dtype=np.int32),
        np.asarray([1, 2, 3], dtype=np.int32),  # world x, world y, yaw
        np.asarray([1.0, 0.0], dtype=np.float64),
        int(model.body("base").id),
        0,
        10,
        0.5,
        0.05,
        float(np.deg2rad(3.0)),
        np.full(3, 1e-6, dtype=np.float64),
        0.0,
    )

    expected_errors = np.asarray(
        [
            0.31622776601683794,
            0.20904575523372224,
            0.10550762017505455,
            0.05287835218818,
            0.02647104188167001,
            0.01324356906246661,
            0.00662380752241153,
            0.00331241159559911,
            0.00165633339648498,
            0.00082819886884547,
            0.000414107606476537,
        ],
        dtype=np.float64,
    )
    expected_qpos = np.asarray(
        [
            0.11077400895609862,
            0.15225167586825236,
            0.0,
            0.9972670075811527,
            0.0,
            0.0,
            0.07388176764353382,
        ],
        dtype=np.float64,
    )
    expected_planar = np.asarray(
        [
            0.11077400895609862,
            0.15225167586825236,
            0.14789829482453773,
        ],
        dtype=np.float64,
    )

    assert tuple(result) == (
        "ik_err_array",
        "idx_best",
        "ik_err_best",
        "elapsed_time",
        "qpos_full_best",
        "qpos_used_best",
        "base_planar_best",
    )
    assert int(result["idx_best"]) == 10
    assert np.isfinite(float(result["elapsed_time"]))
    assert float(result["elapsed_time"]) >= 0.0
    np.testing.assert_allclose(
        np.asarray(result["ik_err_array"]), expected_errors, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        float(result["ik_err_best"]), expected_errors[-1], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result["qpos_full_best"]), expected_qpos, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result["qpos_used_best"]), expected_planar, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result["base_planar_best"]), expected_planar, rtol=0.0, atol=1e-12
    )


def test_compiled_signed_distance_matches_python_with_infinite_radius() -> None:
    backend = resolve_backend("native")
    python_model = MujocoModel.from_robot("k1")
    native_model = MujocoModel.from_robot("k1")
    candidates = build_collision_candidates(
        python_model,
        get_initial_collision_profile("k1"),
    )
    python_model.reset()
    native_model.reset()

    python_batch = query_signed_distances(
        python_model,
        candidates.geom_pairs,
        limit=32,
        distance_limit=np.inf,
        backend="python",
    )
    native_batch = query_signed_distances(
        native_model,
        candidates.geom_pairs,
        limit=32,
        distance_limit=np.inf,
        backend=backend,
    )

    for field_name in (
        "distance",
        "fromto",
        "normal",
        "geom_pairs",
        "body_pairs",
        "source",
        "contact_index",
    ):
        np.testing.assert_array_equal(
            getattr(native_batch, field_name),
            getattr(python_batch, field_name),
        )
