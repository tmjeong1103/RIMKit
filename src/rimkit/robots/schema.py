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
    # Optional public serialization order for articulated scalar joints.  It
    # is used by retarget-only models whose kinematic tree is re-rooted for IK
    # while exported motions must retain the vendor model's joint order.
    export_joint_names: tuple[str, ...] = ()
    # Optional vendor root body for those re-rooted models. Exported motions
    # give the root pose of this body, so they drive the vendor model directly.
    export_root_body: str = ""

    @property
    def actuated_dof(self) -> int:
        """Number of articulated DOFs exposed to retargeting clients.

        Most bundled models provide one actuator per articulated DOF. Models
        such as Asimov-1 intentionally distribute no MuJoCo actuators, so the
        non-floating velocity dimensions are the accurate public DOF count.
        """

        return self.expected_nu if self.expected_nu > 0 else self.expected_nv - 6
