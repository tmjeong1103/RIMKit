"""Static and MuJoCo-backed validation for bundled robot assets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from core_retarget.assets import root_path
from core_retarget.robots.joi import get_body_joi_mapping
from core_retarget.robots.schema import RobotSpec


@dataclass(frozen=True)
class VerificationIssue:
    """One model verification diagnostic."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ModelVerification:
    """Result of checking one robot model contract."""

    robot_id: str
    issues: tuple[VerificationIssue, ...]
    model_info: dict[str, int]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _named_objects(mujoco: Any, model: Any, object_type: Any, count: int) -> set[str]:
    names: set[str] = set()
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        if name is not None:
            names.add(name)
    return names


def verify_robot(spec: RobotSpec, *, load_mujoco: bool = True) -> ModelVerification:
    """Verify files, provenance, hashes, and optionally the compiled MJCF model."""

    assets = root_path()
    model_path = assets / spec.model_relpath
    scene_path = assets / spec.scene_relpath
    issues: list[VerificationIssue] = []
    model_info: dict[str, int] = {}

    required_files = {
        "model": model_path,
        "scene": scene_path,
        "license": assets / spec.license_relpath,
        "source_manifest": assets / spec.source_manifest_relpath,
    }
    for label, path in required_files.items():
        if not path.is_file():
            issues.append(
                VerificationIssue("error", f"missing_{label}", f"Missing {label}: {path}")
            )

    if model_path.is_file():
        try:
            ElementTree.parse(model_path)
        except ElementTree.ParseError as exc:
            issues.append(
                VerificationIssue("error", "invalid_model_xml", f"Invalid model XML: {exc}")
            )
        actual_hash = _digest(model_path)
        if actual_hash != spec.model_sha256:
            issues.append(
                VerificationIssue(
                    "error",
                    "model_hash_mismatch",
                    f"Expected model SHA-256 {spec.model_sha256}, found {actual_hash}.",
                )
            )

    if scene_path.is_file():
        try:
            ElementTree.parse(scene_path)
        except ElementTree.ParseError as exc:
            issues.append(
                VerificationIssue("error", "invalid_scene_xml", f"Invalid scene XML: {exc}")
            )

    if not load_mujoco or not scene_path.is_file():
        return ModelVerification(spec.robot_id, tuple(issues), model_info)

    try:
        import mujoco  # type: ignore[import-untyped]
    except ImportError:
        issues.append(
            VerificationIssue(
                "error",
                "mujoco_unavailable",
                "MuJoCo is not installed; install the core project dependencies.",
            )
        )
        return ModelVerification(spec.robot_id, tuple(issues), model_info)

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    except Exception as exc:
        issues.append(
            VerificationIssue("error", "mujoco_compile_failed", f"MuJoCo load failed: {exc}")
        )
        return ModelVerification(spec.robot_id, tuple(issues), model_info)

    model_info.update(
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        njnt=int(model.njnt),
        nbody=int(model.nbody),
        nsite=int(model.nsite),
    )
    for field, expected in (
        ("nq", spec.expected_nq),
        ("nv", spec.expected_nv),
        ("nu", spec.expected_nu),
    ):
        actual = model_info[field]
        if actual != expected:
            issues.append(
                VerificationIssue(
                    "error",
                    f"{field}_mismatch",
                    f"Expected {field}={expected}, found {actual}.",
                )
            )

    body_names = _named_objects(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_names = _named_objects(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_names = _named_objects(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)

    for label, required, available in (
        ("body", spec.required_bodies, body_names),
        ("joint", spec.required_joints, joint_names),
        ("site", spec.required_sites, site_names),
    ):
        missing = sorted(set(required) - available)
        if missing:
            issues.append(
                VerificationIssue(
                    "error",
                    f"missing_{label}s",
                    f"Missing required {label} names: {', '.join(missing)}.",
                )
            )

    missing_joi_bodies = sorted(set(get_body_joi_mapping(spec.robot_id).values()) - body_names)
    if missing_joi_bodies:
        issues.append(
            VerificationIssue(
                "error",
                "missing_joi_bodies",
                "JOI mapping references missing bodies: " + ", ".join(missing_joi_bodies) + ".",
            )
        )

    free_joint_count = sum(
        int(joint_type == mujoco.mjtJoint.mjJNT_FREE) for joint_type in model.jnt_type
    )
    if free_joint_count != 1:
        issues.append(
            VerificationIssue(
                "error",
                "floating_base_count",
                f"Expected one free joint, found {free_joint_count}.",
            )
        )

    return ModelVerification(spec.robot_id, tuple(issues), model_info)
