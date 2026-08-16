"""Small, distribution-safe MuJoCo model adapter used by retargeting stages."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rimkit.assets import root_path
from rimkit.exceptions import ModelVerificationError
from rimkit.robots.joi import get_body_joi_mapping
from rimkit.robots.registry import get_robot
from rimkit.robots.schema import RobotSpec


def _load_mujoco() -> Any:
    """Import MuJoCo only when a model is actually constructed."""

    try:
        return import_module("mujoco")
    except ImportError as exc:
        raise ModelVerificationError(
            "MuJoCo is required to load a robot model. Install the core project "
            "dependencies before running retargeting."
        ) from exc


def _joint_width(joint_type: int, mujoco: Any, *, qpos: bool) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7 if qpos else 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4 if qpos else 3
    return 1


class MujocoModel:
    """A validated MuJoCo scene and its mutable kinematic state.

    Use :meth:`from_robot` for public robots.  Passing ``robot_spec`` to the
    constructor is useful for tests or other callers that already resolved the
    registry entry; the same model-contract checks are applied either way.
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        robot_spec: RobotSpec | None = None,
    ) -> None:
        self.path = Path(xml_path).expanduser().resolve()
        self.robot_spec = robot_spec
        if not self.path.is_file():
            raise ModelVerificationError(f"MuJoCo XML does not exist: {self.path}")

        self._mujoco: Any = _load_mujoco()
        try:
            self.model: Any = self._mujoco.MjModel.from_xml_path(str(self.path))
            self.data: Any = self._mujoco.MjData(self.model)
        except Exception as exc:
            raise ModelVerificationError(
                f"Could not compile MuJoCo XML {self.path}: {exc}"
            ) from exc

        body_names = self._object_names(self._mujoco.mjtObj.mjOBJ_BODY, int(self.model.nbody))
        self.joint_names = self._object_names(self._mujoco.mjtObj.mjOBJ_JOINT, int(self.model.njnt))
        if any(name is None for name in body_names):
            unnamed_ids = [index for index, name in enumerate(body_names) if name is None]
            raise ModelVerificationError(
                f"MuJoCo bodies at indices {unnamed_ids} have no names in {self.path}."
            )
        # Narrowing a tuple element-by-element is not represented by mypy, so
        # construct the public all-string tuple after the explicit check.
        self.body_names = tuple(str(name) for name in body_names)
        self._body_ids = {name: index for index, name in enumerate(self.body_names)}
        self._joint_ids = {
            name: index for index, name in enumerate(self.joint_names) if name is not None
        }
        self.qpos_joint_names = self._qpos_component_names()

        hinge = int(self._mujoco.mjtJoint.mjJNT_HINGE)
        slide = int(self._mujoco.mjtJoint.mjJNT_SLIDE)
        self.rev_joint_names = tuple(
            name
            for index, name in enumerate(self.joint_names)
            if name is not None and int(self.model.jnt_type[index]) == hinge
        )
        self.pri_joint_names = tuple(
            name
            for index, name in enumerate(self.joint_names)
            if name is not None and int(self.model.jnt_type[index]) == slide
        )
        # Preserve solver ordering: all revolute joints, followed by
        # all prismatic joints (not an interleaved model order).
        self.rev_pri_joint_names = self.rev_joint_names + self.pri_joint_names

        q0 = np.array(self.model.qpos0, dtype=np.float64, copy=True)
        self._validate_contract(q0)
        q0.setflags(write=False)
        self.q0: NDArray[np.float64] = q0
        self.reset()

    def _qpos_component_names(self) -> tuple[str, ...]:
        """Return one stable name for each non-root scalar in MuJoCo qpos order."""

        free_type = int(self._mujoco.mjtJoint.mjJNT_FREE)
        ball_type = int(self._mujoco.mjtJoint.mjJNT_BALL)
        names: list[str] = []
        for index, name in enumerate(self.joint_names):
            joint_type = int(self.model.jnt_type[index])
            if joint_type == free_type:
                continue
            if name is None:
                raise ModelVerificationError(
                    f"Non-root MuJoCo joint at index {index} has no name in {self.path}."
                )
            if joint_type == ball_type:
                names.extend(f"{name}_quat_{axis}" for axis in ("w", "x", "y", "z"))
            else:
                names.append(name)
        return tuple(names)

    @classmethod
    def from_robot(cls, robot_id: str) -> MujocoModel:
        """Load and validate one registry-selected packaged robot scene."""

        spec = get_robot(robot_id)
        return cls(root_path() / spec.scene_relpath, robot_spec=spec)

    def _object_names(self, object_type: Any, count: int) -> tuple[str | None, ...]:
        names: list[str | None] = []
        for index in range(count):
            name = self._mujoco.mj_id2name(self.model, object_type, index)
            names.append(None if name is None else str(name))
        return tuple(names)

    def _validate_contract(self, q0: NDArray[np.float64]) -> None:
        spec = self.robot_spec
        if spec is not None:
            dimensions = (
                ("nq", int(self.model.nq), spec.expected_nq),
                ("nv", int(self.model.nv), spec.expected_nv),
                ("nu", int(self.model.nu), spec.expected_nu),
            )
            mismatches = [
                f"{name}={actual} (expected {expected})"
                for name, actual, expected in dimensions
                if actual != expected
            ]
            if mismatches:
                raise ModelVerificationError(
                    f"Robot {spec.robot_id!r} model dimensions do not match the registry: "
                    + ", ".join(mismatches)
                    + "."
                )

            mapped_bodies = set(get_body_joi_mapping(spec.robot_id).values())
            missing_bodies = sorted(mapped_bodies.difference(self._body_ids))
            if missing_bodies:
                raise ModelVerificationError(
                    f"Robot {spec.robot_id!r} JOI mapping references missing bodies: "
                    + ", ".join(missing_bodies)
                    + "."
                )

        free_type = int(self._mujoco.mjtJoint.mjJNT_FREE)
        free_joint_ids = [
            index
            for index in range(int(self.model.njnt))
            if int(self.model.jnt_type[index]) == free_type
        ]
        if len(free_joint_ids) != 1:
            raise ModelVerificationError(
                f"Expected exactly one floating-base joint; found {len(free_joint_ids)}."
            )

        if q0.shape != (int(self.model.nq),) or not np.isfinite(q0).all():
            raise ModelVerificationError(
                f"Model qpos0 must be a finite vector of shape ({int(self.model.nq)},)."
            )

        free_qpos_address = int(self.model.jnt_qposadr[free_joint_ids[0]])
        quaternion = q0[free_qpos_address + 3 : free_qpos_address + 7]
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-12:
            raise ModelVerificationError(
                "The floating-base quaternion in model qpos0 is not valid."
            )

    def reset(self) -> None:
        """Reset simulation data to the model home pose and run kinematics."""

        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q0
        self._mujoco.mj_forward(self.model, self.data)

    def forward(
        self,
        qpos: NDArray[np.floating[Any]] | Sequence[float] | None = None,
        *,
        clip_position: bool = True,
    ) -> None:
        """Optionally set a full qpos vector and update forward kinematics."""

        if qpos is not None:
            values = np.asarray(qpos, dtype=np.float64)
            expected_shape = (int(self.model.nq),)
            if values.shape != expected_shape:
                raise ValueError(f"qpos must have shape {expected_shape}; found {values.shape}.")
            if not np.isfinite(values).all():
                raise ValueError("qpos contains NaN or infinity.")
            self.data.qpos[:] = values

        if clip_position:
            for joint_name in self.rev_pri_joint_names:
                joint_id = self._joint_ids[joint_name]
                if not bool(self.model.jnt_limited[joint_id]):
                    continue
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                lower, upper = self.model.jnt_range[joint_id]
                self.data.qpos[qpos_address] = np.clip(
                    self.data.qpos[qpos_address], float(lower), float(upper)
                )

        self._mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _normalize_names(names: Sequence[str] | str) -> tuple[str, ...]:
        return (names,) if isinstance(names, str) else tuple(names)

    def _joint_id(self, name: str) -> int:
        try:
            return self._joint_ids[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MuJoCo joint {name!r}.") from exc

    def get_qpos(self, joint_names: Sequence[str] | str | None = None) -> NDArray[np.float64]:
        """Return a defensive copy of full or selected joint positions."""

        if joint_names is None:
            return np.array(self.data.qpos, dtype=np.float64, copy=True)

        parts: list[NDArray[np.float64]] = []
        for name in self._normalize_names(joint_names):
            joint_id = self._joint_id(name)
            address = int(self.model.jnt_qposadr[joint_id])
            width = _joint_width(int(self.model.jnt_type[joint_id]), self._mujoco, qpos=True)
            parts.append(np.array(self.data.qpos[address : address + width], copy=True))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def get_qpos_indices(self, names: Sequence[str] | str) -> NDArray[np.int32]:
        """Return each joint's first qpos address in caller-provided order."""

        normalized_names = self._normalize_names(names)
        return np.asarray(
            [int(self.model.jnt_qposadr[self._joint_id(name)]) for name in normalized_names],
            dtype=np.int32,
        )

    def get_dof_indices(self, names: Sequence[str] | str) -> NDArray[np.int32]:
        """Return Jacobian DOF columns, expanding free and ball joints."""

        indices: list[int] = []
        for name in self._normalize_names(names):
            joint_id = self._joint_id(name)
            address = int(self.model.jnt_dofadr[joint_id])
            width = _joint_width(int(self.model.jnt_type[joint_id]), self._mujoco, qpos=False)
            indices.extend(range(address, address + width))
        return np.asarray(indices, dtype=np.int32)

    def get_body_transform(self, name: str) -> NDArray[np.float64]:
        """Return one body's current world transform as a new 4x4 matrix."""

        try:
            body_id = self._body_ids[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MuJoCo body {name!r}.") from exc

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body_id]
        return transform

    def copy_state_from(self, other: MujocoModel) -> None:
        """Copy compatible dynamic state from another adapter and run FK."""

        dimensions = ("nq", "nv", "nu", "nbody", "nmocap")
        mismatches = [
            name
            for name in dimensions
            if int(getattr(self.model, name)) != int(getattr(other.model, name))
        ]
        if mismatches or self.joint_names != other.joint_names:
            details = ", ".join(mismatches) if mismatches else "joint ordering"
            raise ValueError(f"Cannot copy state between incompatible models ({details}).")

        self.data.time = float(other.data.time)
        for field in (
            "qpos",
            "qvel",
            "qacc",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "act",
            "qacc_warmstart",
            "userdata",
            "eq_active",
        ):
            destination = getattr(self.data, field, None)
            source = getattr(other.data, field, None)
            if destination is not None and source is not None and destination.shape == source.shape:
                destination[:] = source
        self._mujoco.mj_forward(self.model, self.data)


__all__ = ["MujocoModel"]
