"""Immutable descriptions of supported robot assets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    """Static contract for one bundled MuJoCo robot."""

    robot_id: str
    display_name: str
    manufacturer: str
    model_relpath: str
    scene_relpath: str
    license_spdx: str
    license_relpath: str
    source_repository: str
    source_revision: str
    source_manifest_relpath: str
    model_sha256: str
    expected_nq: int
    expected_nv: int
    expected_nu: int
    required_bodies: tuple[str, ...]
    required_joints: tuple[str, ...]
    required_sites: tuple[str, ...] = ()
    compatibility_joints: tuple[str, ...] = ()

    @property
    def actuated_dof(self) -> int:
        """Number of actuators in the distributed model contract."""

        return self.expected_nu
