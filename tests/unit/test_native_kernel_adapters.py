from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np

from rimkit.mujoco.collision import query_signed_distances
from rimkit.mujoco.ik import BodyPositionIKSolver
from rimkit.mujoco.model import MujocoModel
from rimkit.native import BackendSelection

_MODEL_XML = """
<mujoco model="native-adapter-test">
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


def _selection(module: ModuleType) -> BackendSelection:
    return BackendSelection(
        requested="native",
        selected="native",
        reason="test",
        module_name="test.native",
        _module=module,
    )


def test_ik_adapter_passes_c_contiguous_frozen_binding_contract() -> None:
    observed: list[tuple[Any, ...]] = []
    module = ModuleType("test_native_ik")

    def solve(*args: Any) -> dict[str, Any]:
        observed.append(args)
        max_iterations = int(args[26])
        qpos = np.asarray([0.25], dtype=np.float64)
        return {
            "q_rev_pri_best": qpos,
            "ik_err_array": np.linspace(1.0, 0.0, max_iterations + 1),
            "idx_best": max_iterations,
            "ik_err_best": 0.0,
            "elapsed_time": 0.001,
            "qpos_full_best": qpos,
            "qpos_used_best": qpos,
        }

    module.solve_body_position_ik = solve  # type: ignore[attr-defined]
    adapter = _TinyAdapter()
    solver = BodyPositionIKSolver(
        adapter,
        max_iterations=2,
        backend=_selection(module),
    )
    current = adapter.data.xpos[adapter.model.body("tip").id].copy()
    solver.add_target("tip", current, current + np.asarray([0.0, 0.1, 0.0]))

    result = solver.solve()

    assert result.best_iteration == 2
    assert result.iterations == 2
    np.testing.assert_array_equal(result.qpos, np.asarray([0.25]))
    assert len(observed) == 1
    arguments = observed[0]
    assert len(arguments) == 37
    assert arguments[2].shape == (1,)
    assert arguments[3].shape == (1, 3)
    assert arguments[4].shape == (1, 3)
    assert arguments[5].shape == (1,)
    assert arguments[8].dtype == np.dtype(np.int32)
    assert arguments[14].shape == (1,)
    assert all(
        not isinstance(value, np.ndarray) or value.flags.c_contiguous for value in arguments[2:21]
    )


def test_signed_distance_adapter_converts_native_arrays_to_read_only_contract() -> None:
    observed: list[tuple[Any, ...]] = []
    module = ModuleType("test_native_distance")

    def query(*args: Any) -> dict[str, Any]:
        observed.append(args)
        return {
            "dist": np.asarray([-0.01]),
            "fromto": np.asarray([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]]),
            "normal": np.asarray([[1.0, 0.0, 0.0]]),
            "geom_pair": np.asarray([[0, 1]], dtype=np.int32),
            "body_pair": np.asarray([[1, 2]], dtype=np.int32),
            "source": np.asarray([1], dtype=np.int32),
            "contact_idx": np.asarray([3], dtype=np.int32),
            "n_result": 1,
        }

    module.signed_distance_arrays = query  # type: ignore[attr-defined]
    model = SimpleNamespace(ngeom=2, _address=101)
    data = SimpleNamespace(_address=202)
    adapter = cast(MujocoModel, SimpleNamespace(model=model, data=data))

    batch = query_signed_distances(
        adapter,
        np.asarray([[0, 1]], dtype=np.int32),
        limit=4,
        backend=_selection(module),
    )

    assert len(batch) == 1
    assert batch.distance[0] == -0.01
    assert not batch.distance.flags.writeable
    assert observed[0][0:2] == (101, 202)
    assert observed[0][4] == 4
