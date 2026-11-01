from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rimkit.exceptions import MotionValidationError
from rimkit.motion.bvh import (
    _rotation_offsets,
    convert_bones_seed_bvh,
    estimate_bones_seed_contacts,
    load_bones_seed_bvh,
)
from rimkit.motion.soma import load_soma_motion
from rimkit.motion.soma_joints import SOMA77_JOINT_NAMES, SOMA77_JOINT_PARENTS


def test_estimated_contacts_distinguish_rest_flight_and_sliding():
    positions = np.zeros((90, 77, 3))
    indices = [
        SOMA77_JOINT_NAMES.index(name)
        for name in (
            "LeftFoot",
            "LeftToeBase",
            "LeftToeEnd",
            "RightFoot",
            "RightToeBase",
            "RightToeEnd",
        )
    ]
    static, provenance = estimate_bones_seed_contacts(positions, 30.0)
    assert static.shape == (90, 6) and static.all()
    assert "not force measurements" in provenance["semantics"]
    # An actual flight interval must stay unsupported even when both feet stop
    # moving at its apex; height and speed are independent conditions.
    positions[30:60, indices, 1] = 0.3
    flight, _ = estimate_bones_seed_contacts(positions, 30.0)
    assert not flight[40:50].any()
    assert flight[:20].all() and flight[70:].all()
    # A foot sliding across the floor is not a stationary support point.
    positions[:] = 0.0
    positions[30:60, indices[:3], 0] = np.linspace(0, 1, 30)[:, None]
    positions[60:, indices[:3], 0] = 1.0
    sliding, _ = estimate_bones_seed_contacts(positions, 30.0)
    assert not sliding[40:50, :3].any()
    assert sliding[:, 3:].all()
    # A foot held up for the whole clip never defines its own floor.
    positions[:] = 0.0
    positions[:, indices[:3], 1] = 0.3
    raised, _ = estimate_bones_seed_contacts(positions, 30.0)
    assert not raised[:, :3].any()
    assert raised[:, 3:].all()


def test_estimated_contact_provenance_survives_conversion_and_schedule(tmp_path):
    from rimkit.motion.contacts import build_contact_schedule

    source = _soma_bvh(tmp_path / "source.bvh")
    output = convert_bones_seed_bvh(
        source, tmp_path / "contacts.npz", target_fps=8, estimate_contacts=True
    )
    with np.load(output, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["foot_contact_provenance"].item()))
        assert archive["foot_contacts"].shape == (9, 6)
        assert provenance["channel_names"][1:3] == ["LeftToeBase", "LeftToeEnd"]
        assert len(provenance["source_bvh_sha256"]) == 64
    motion = load_soma_motion(output)
    assert motion.foot_contact_source == "bones_seed_kinematic_foot_point_contacts_v1"
    schedule = build_contact_schedule(motion)
    assert schedule.contact_source.startswith("bones_seed_kinematic_foot_point_contacts_v1+")


def test_named_contacts_cannot_omit_provenance(tmp_path):
    source = _soma_bvh(tmp_path / "source.bvh")
    valid = convert_bones_seed_bvh(source, tmp_path / "valid.npz", estimate_contacts=True)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {
            name: archive[name] for name in archive.files if name != "foot_contact_provenance"
        }
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **arrays)
    with pytest.raises(MotionValidationError, match="JSON provenance"):
        load_soma_motion(invalid)


def _soma_bvh(path: Path, *, bad_parent: bool = False) -> Path:
    """Encode a known rigid Z rotation in original SEED joint frames."""
    children: dict[str, list[str]] = {"Root": ["Hips"]}
    for name, parent in SOMA77_JOINT_PARENTS:
        children.setdefault(name, [])
        if parent is not None:
            children.setdefault(parent, []).append(name)
    if bad_parent:
        children["Hips"].remove("LeftLeg")
        children["Spine1"].append("LeftLeg")
    lines = ["HIERARCHY"]
    traversal = []

    def emit(name: str) -> None:
        traversal.append(name)
        tag = "ROOT" if name == "Root" else "JOINT"
        lines.extend([f"{tag} {name}", "{", "OFFSET 100 0 0"])
        if name in {"Root", "Hips"}:
            lines.append("CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append("CHANNELS 3 Zrotation Yrotation Xrotation")
        for child in children[name]:
            emit(child)
        if not children[name]:
            lines.extend(["End Site", "{", "OFFSET 0 0 0", "}"])
        lines.append("}")

    emit("Root")
    lines.extend(["MOTION", "Frames: 5", "Frame Time: 0.25"])
    offset_map = dict(zip(SOMA77_JOINT_NAMES, _rotation_offsets(), strict=True))
    for frame in range(5):
        world = {"Root": np.eye(3)}
        standard = Rotation.from_euler("Z", frame * 45, degrees=True).as_matrix()
        values = []
        for name in traversal:
            if name == "Root":
                local_rotation = np.eye(3)
                values.extend([10, 20, 30])
            else:
                parent = dict(SOMA77_JOINT_PARENTS)[name] or "Root"
                world[name] = standard @ offset_map[name]
                local_rotation = world[parent].T @ world[name]
                if name == "Hips":
                    values.extend([frame * 100, 200, 0])
            values.extend(Rotation.from_matrix(local_rotation).as_euler("ZYX", degrees=True))
        lines.append(" ".join(str(float(value)) for value in values))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_absolute_translation_and_root_rotation(tmp_path: Path) -> None:
    path = tmp_path / "simple.bvh"
    path.write_text("""HIERARCHY
ROOT Root
{
OFFSET 999 999 999
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Hips
{
OFFSET 999 999 999
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Tip
{
OFFSET 1 2 3
CHANNELS 3 Zrotation Yrotation Xrotation
End Site
{
OFFSET 0 0 0
}
}
}
}
MOTION
Frames: 1
Frame Time: 0.25
10 0 0 90 0 0 2 3 0 0 90 0 0 0 0
""")
    motion = load_bones_seed_bvh(path)
    positions, rotations = motion.global_transforms()
    assert motion.names == ("Root", "Hips", "Tip")
    np.testing.assert_allclose(positions[0], [[10, 0, 0], [7, 2, 0], [5, 5, -1]])
    np.testing.assert_allclose(rotations[0, 1, :, 0], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(rotations[0, 1, :, 2], [0, 1, 0], atol=1e-12)


def test_conversion_joint_axes_slerp_and_timing(tmp_path: Path) -> None:
    source = _soma_bvh(tmp_path / "source.bvh")
    output = convert_bones_seed_bvh(source, tmp_path / "converted.npz", target_fps=8)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["posed_joints"].shape == (9, 77, 3)
        np.testing.assert_allclose(
            archive["root_positions"][[0, 4, 8]],
            [[0.1, 2.2, 0.3], [2.1, 2.2, 0.3], [4.1, 2.2, 0.3]],
        )
        expected = Rotation.from_euler(
            "Z", (np.arange(9) * 22.5)[:, None], degrees=True
        ).as_matrix()
        np.testing.assert_allclose(
            archive["global_rot_mats"], np.tile(expected[:, None], (1, 77, 1, 1)), atol=1e-6
        )
        assert "foot_contacts" not in archive.files
        assert all(not archive[key].dtype.hasobject for key in archive.files)
        metadata = json.loads(str(archive["conversion_metadata"]))
        assert metadata["source_frames"] == 5
        assert metadata["source_fps"] == 4
    loaded = load_soma_motion(output)
    assert loaded.seconds[-1] == 1
    np.testing.assert_allclose(loaded.posed_joints[0, 0], [0.1, -0.3, 2.2])


def test_source_bone_lengths_and_grounding_are_preserved(tmp_path: Path) -> None:
    source = _soma_bvh(tmp_path / "source.bvh")
    raw = convert_bones_seed_bvh(source, tmp_path / "raw.npz")
    grounded = convert_bones_seed_bvh(source, tmp_path / "grounded.npz", ground=True)
    with np.load(raw) as a, np.load(grounded) as b:
        positions = a["posed_joints"]
        for name, parent in SOMA77_JOINT_PARENTS[1:]:
            length = np.linalg.norm(
                positions[:, SOMA77_JOINT_NAMES.index(name)]
                - positions[:, SOMA77_JOINT_NAMES.index(parent)],
                axis=1,
            )
            np.testing.assert_allclose(length, 1, atol=1e-12)
        floor = json.loads(str(b["conversion_metadata"]))["floor_y_subtracted"]
        np.testing.assert_allclose(b["posed_joints"], positions - [0, floor, 0])


def test_invalid_topology_and_overwrite_are_rejected(tmp_path: Path) -> None:
    source = _soma_bvh(tmp_path / "source.bvh", bad_parent=True)
    with pytest.raises(MotionValidationError, match="parent for LeftLeg"):
        convert_bones_seed_bvh(source, tmp_path / "out.npz")
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"keep")
    with pytest.raises(MotionValidationError, match="overwrite"):
        convert_bones_seed_bvh(source, existing)
    assert existing.read_bytes() == b"keep"


@pytest.mark.parametrize("replacement", ["nan", "inf", "invalid"])
def test_invalid_frame_data_is_rejected(tmp_path: Path, replacement: str) -> None:
    path = _soma_bvh(tmp_path / "source.bvh")
    lines = path.read_text().splitlines()
    last = lines[-1].split()
    last[0] = replacement
    lines[-1] = " ".join(last)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MotionValidationError):
        load_bones_seed_bvh(path)


def test_failed_write_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _soma_bvh(tmp_path / "source.bvh")
    output = tmp_path / "output.npz"

    def fail_write(stream: object, **arrays: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="No space"):
        convert_bones_seed_bvh(source, output)
    assert not output.exists()
    assert list(tmp_path.glob("*.npz")) == []


def test_frame_limit_is_checked_before_resampling(tmp_path: Path) -> None:
    source = _soma_bvh(tmp_path / "source.bvh")
    # Five frames 0.25 s apart resample to 31 frames at 30 FPS.
    convert_bones_seed_bvh(source, tmp_path / "fits.npz", target_fps=30, max_frames=31)
    source.write_text(source.read_text().replace("Frame Time: 0.25", "Frame Time: 250"))
    output = tmp_path / "stretched.npz"
    with pytest.raises(MotionValidationError, match="30001 frames; the limit is 1800"):
        convert_bones_seed_bvh(source, output, target_fps=30, max_frames=1800)
    assert not output.exists()


def test_deeply_nested_hierarchy_is_rejected(tmp_path: Path) -> None:
    depth = 5000
    body = "{\nOFFSET 0 0 0\nCHANNELS 3 Zrotation Yrotation Xrotation\n"
    nested = "".join(f"JOINT j{index}\n{body}" for index in range(1, depth))
    path = tmp_path / "deep.bvh"
    path.write_text(f"HIERARCHY\nROOT j0\n{body}{nested}" + "}\n" * depth + "MOTION\nFrames: 1\n")
    with pytest.raises(MotionValidationError, match="Malformed BVH"):
        load_bones_seed_bvh(path)
