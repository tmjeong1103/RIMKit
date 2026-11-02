"""Convert BONES-SEED SOMA BVH to the evaluated SOMA77 NPZ contract.

This adapter follows BONES-SEED's absolute local position channels: a position
channel replaces that axis of OFFSET. It is not a generic BVH dialect adapter.
FK uses the BVH bone lengths, including for SOMA Proportional. The packaged
Kimodo rotation offsets change joint-local axes without changing positions.
The output remains Y-up in meters; RIMKit's NPZ loader performs Z-up conversion.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, median_filter  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation, Slerp  # type: ignore[import-untyped]

from rimkit.assets import root_path
from rimkit.exceptions import MotionValidationError
from rimkit.motion.soma import validate_soma_npz
from rimkit.motion.soma_joints import SOMA77_JOINT_NAMES, SOMA77_JOINT_PARENTS

FloatArray = NDArray[np.float64]


def estimate_bones_seed_contacts(
    positions: FloatArray,
    fps: float,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Estimate six named foot-point contacts from Y-up source kinematics.

    These are source-derived support estimates, not measured forces or human
    annotations. A per-point low quantile is the flat-floor height proxy, which
    accounts for ankle/toe joints having different rest heights. Both feet share
    one floor, so a point's floor is capped just above its mirrored point's; a
    foot that never lands is then not labelled as resting. ToeBase/ToeEnd
    channels are used by RIMKit; the ankle channels retain their own estimates.
    """
    names = ("LeftFoot", "LeftToeBase", "LeftToeEnd", "RightFoot", "RightToeBase", "RightToeEnd")
    indices = [SOMA77_JOINT_NAMES.index(name) for name in names]
    points = np.asarray(positions, dtype=np.float64)[:, indices]
    if len(points) < 2 or not np.isfinite(points).all() or not np.isfinite(fps) or fps <= 0:
        raise MotionValidationError("Contact estimation needs >=2 finite frames and positive FPS.")
    sigma_s, height_m, speed_mps, quantile, mirror_m = 0.025, 0.035, 0.25, 0.02, 0.05
    smoothed = gaussian_filter1d(points, sigma=sigma_s * fps, axis=0, mode="nearest")
    speed = np.linalg.norm(np.gradient(smoothed, 1.0 / fps, axis=0), axis=2)
    floor = np.quantile(smoothed[:, :, 1], quantile, axis=0)
    floor = np.minimum(floor, floor[[3, 4, 5, 0, 1, 2]] + mirror_m)
    raw = (smoothed[:, :, 1] - floor <= height_m) & (speed <= speed_mps)
    window = max(1, int(round(0.05 * fps)))
    if window % 2 == 0:
        window += 1
    labels = median_filter(raw.astype(np.uint8), size=(window, 1), mode="nearest").astype(bool)
    provenance = {
        "algorithm": "bones_seed_kinematic_foot_point_contacts_v1",
        "algorithm_version": 1,
        "semantics": "estimated source support; not force measurements or human annotations",
        "coordinate_system": "source Y-up meters",
        "channel_names": list(names),
        "floor_method": (
            "per-point low quantile on the complete flat-floor clip, "
            "at most floor_mirror_margin_m above the mirrored point's"
        ),
        "floor_quantile": quantile,
        "floor_mirror_margin_m": mirror_m,
        "floor_y_m": floor.tolist(),
        "height_threshold_m": height_m,
        "speed_threshold_mps": speed_mps,
        "smoothing_sigma_s": sigma_s,
        "median_window_frames": window,
        "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return labels, provenance


@dataclass(frozen=True)
class BvhMotion:
    """Parsed joints in hierarchy order, excluding BVH End Site markers."""

    names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: FloatArray
    channels: tuple[tuple[str, ...], ...]
    values: FloatArray
    frame_time: float

    def global_transforms(self) -> tuple[FloatArray, FloatArray]:
        """Evaluate BONES-SEED local translations and intrinsic Euler rotations."""

        frame_count = len(self.values)
        positions = np.empty((frame_count, len(self.names), 3), dtype=np.float64)
        rotations = np.empty((frame_count, len(self.names), 3, 3), dtype=np.float64)
        cursor = 0
        for joint, channels in enumerate(self.channels):
            local_position = np.tile(self.offsets[joint], (frame_count, 1))
            angles = []
            axes = []
            for channel in channels:
                if channel.endswith("position"):
                    # SEED/SOMA translations already include the rest offset.
                    local_position[:, "XYZ".index(channel[0])] = self.values[:, cursor]
                else:
                    axes.append(channel[0])
                    angles.append(self.values[:, cursor])
                cursor += 1
            local_rotation = Rotation.from_euler(
                "".join(axes), np.stack(angles, axis=1), degrees=True
            ).as_matrix()
            parent = self.parents[joint]
            if parent < 0:
                positions[:, joint] = local_position
                rotations[:, joint] = local_rotation
            else:
                positions[:, joint] = positions[:, parent] + np.einsum(
                    "tij,tj->ti", rotations[:, parent], local_position
                )
                rotations[:, joint] = rotations[:, parent] @ local_rotation
        return positions, rotations


def load_bones_seed_bvh(path: str | Path) -> BvhMotion:
    """Parse a SOMA BVH, rejecting malformed or incomplete motion data."""

    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise MotionValidationError("BVH must be UTF-8 text.") from error
    sections = re.split(r"\bMOTION\b", text, maxsplit=1)
    if len(sections) != 2:
        raise MotionValidationError("BVH must contain a MOTION section.")
    tokens = iter(re.findall(r"[{}]|[^\s{}]+", sections[0]))
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[tuple[str, ...]] = []

    def expect(value: str) -> None:
        actual = next(tokens)
        if actual != value:
            raise MotionValidationError(f"Expected BVH token {value!r}; found {actual!r}.")

    def joint(parent: int) -> None:
        name = next(tokens)
        if name in names:
            raise MotionValidationError(f"Duplicate BVH joint {name!r}.")
        index = len(names)
        names.append(name)
        parents.append(parent)
        expect("{")
        expect("OFFSET")
        offsets.append([float(next(tokens)) for _ in range(3)])
        expect("CHANNELS")
        count = int(next(tokens))
        layout = tuple(next(tokens) for _ in range(count))
        rotation_channels = tuple(c for c in layout if c.endswith("rotation"))
        position_channels = tuple(c for c in layout if c.endswith("position"))
        if (
            count not in (3, 6)
            or len(set(layout)) != count
            or set(rotation_channels) != {"Xrotation", "Yrotation", "Zrotation"}
            or (count == 6 and set(position_channels) != {"Xposition", "Yposition", "Zposition"})
            or layout != position_channels + rotation_channels
        ):
            raise MotionValidationError(f"Unsupported BVH channels for {name}: {layout}.")
        channels.append(layout)
        while True:
            token = next(tokens)
            if token == "}":
                break
            if token == "JOINT":
                joint(index)
            elif token == "End":
                expect("Site")
                expect("{")
                expect("OFFSET")
                for _ in range(3):
                    if not np.isfinite(float(next(tokens))):
                        raise MotionValidationError("BVH End Site offset is not finite.")
                expect("}")
            else:
                raise MotionValidationError(f"Unexpected BVH token {token!r}.")

    try:
        expect("HIERARCHY")
        expect("ROOT")
        joint(-1)
        if next(tokens, None) is not None:
            raise MotionValidationError("Unexpected tokens after BVH hierarchy.")
        header = re.fullmatch(
            r"\s*Frames:\s*(\d+)\s+Frame\s+Time:\s*(\S+)\s*\n(.*)",
            sections[1],
            flags=re.DOTALL,
        )
        if header is None:
            raise MotionValidationError("Invalid BVH frame count or frame time header.")
        frame_count = int(header[1])
        frame_time = float(header[2])
        if frame_count < 1 or not np.isfinite(frame_time) or frame_time <= 0:
            raise MotionValidationError("BVH requires positive frame count and frame time.")
        # Parsing each row also detects a malformed row with the correct total size.
        rows = [line.split() for line in header[3].splitlines() if line.strip()]
        channel_count = sum(map(len, channels))
        if len(rows) != frame_count or any(len(row) != channel_count for row in rows):
            raise MotionValidationError(
                f"BVH must contain {frame_count} frames of {channel_count} channels."
            )
        values = np.asarray(rows, dtype=np.float64)
        offset_array = np.asarray(offsets, dtype=np.float64)
        if not np.isfinite(values).all() or not np.isfinite(offset_array).all():
            raise MotionValidationError("BVH contains NaN or infinity.")
    except (StopIteration, ValueError, OverflowError, RecursionError) as error:
        raise MotionValidationError(f"Malformed BVH: {error}") from error
    return BvhMotion(
        tuple(names), tuple(parents), offset_array, tuple(channels), values, frame_time
    )


def _soma_indices(motion: BvhMotion) -> list[int]:
    if set(motion.names) not in (set(SOMA77_JOINT_NAMES), set(SOMA77_JOINT_NAMES) | {"Root"}):
        missing = sorted(set(SOMA77_JOINT_NAMES) - set(motion.names))
        extra = sorted(set(motion.names) - set(SOMA77_JOINT_NAMES) - {"Root"})
        raise MotionValidationError(f"BVH is not SOMA77: missing={missing}, extra={extra}.")
    if "Root" in motion.names and motion.parents[motion.names.index("Root")] != -1:
        raise MotionValidationError("The optional Root must be the BVH hierarchy root.")
    indices = [motion.names.index(name) for name in SOMA77_JOINT_NAMES]
    for index, (name, expected_parent) in zip(indices, SOMA77_JOINT_PARENTS, strict=True):
        parent_index = motion.parents[index]
        actual_parent = None if parent_index < 0 else motion.names[parent_index]
        if actual_parent == "Root":
            actual_parent = None
        if actual_parent != expected_parent:
            raise MotionValidationError(f"Unexpected SOMA parent for {name}: {actual_parent}.")
    return indices


def _rotation_offsets() -> FloatArray:
    path = root_path() / "soma" / "soma77_bvh_rotation_offsets.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive["joint_names"].astype(str).tolist())
        offsets = np.asarray(archive["global_rot_offsets"], dtype=np.float64)
    if names != SOMA77_JOINT_NAMES or offsets.shape != (77, 3, 3):
        raise MotionValidationError("Packaged SOMA BVH rotation offsets do not match SOMA77.")
    if (
        not np.isfinite(offsets).all()
        or not np.allclose(offsets @ offsets.transpose(0, 2, 1), np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(offsets), 1.0, atol=1e-6)
    ):
        raise MotionValidationError("Packaged SOMA BVH offsets are not proper rotations.")
    return offsets


def convert_bones_seed_bvh(
    source: str | Path,
    destination: str | Path,
    *,
    target_fps: float | None = None,
    ground: bool = False,
    estimate_contacts: bool = False,
    max_frames: int | None = None,
) -> Path:
    """Write a safe evaluated SOMA77 NPZ, optionally resampling and grounding.

    Position channels are absolute local translations, in centimeters. Rotations
    are changed into Kimodo's standard T-pose frames with G_standard = G_bvh O^T.
    Contacts are absent by default. ``estimate_contacts`` writes explicitly
    identified source kinematic support estimates and their full provenance.
    Resampling interpolates positions and SLERPs rotations at unchanged times.
    ``max_frames`` rejects the motion from its header, before resampling.
    """

    source_path = Path(source).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise MotionValidationError("The converted motion must use an .npz extension.")
    if output.exists():
        raise MotionValidationError(f"Refusing to overwrite {output}.")
    bvh = load_bones_seed_bvh(source_path)
    fps = 1.0 / bvh.frame_time if target_fps is None else float(target_fps)
    if not np.isfinite(fps) or not 0 < fps <= 1000:
        raise MotionValidationError("Target FPS must be finite and in (0, 1000].")
    old_seconds = np.arange(len(bvh.values)) * bvh.frame_time
    frame_count = (
        len(bvh.values) if target_fps is None else int(np.floor(old_seconds[-1] * fps + 1e-9)) + 1
    )
    # A tiny file with a huge Frame Time would otherwise resample to millions of frames.
    if max_frames is not None and frame_count > max_frames:
        raise MotionValidationError(
            f"BVH converts to {frame_count} frames; the limit is {max_frames}."
        )
    indices = _soma_indices(bvh)
    positions, rotations = bvh.global_transforms()
    positions = positions[:, indices] * 0.01
    offsets = _rotation_offsets()
    rotations = rotations[:, indices] @ offsets.transpose(0, 2, 1)
    if target_fps is not None and len(positions) > 1:
        new_seconds = np.arange(frame_count) / fps
        resampled = np.empty((len(new_seconds), 77, 3), dtype=np.float64)
        new_rotations = np.empty((len(new_seconds), 77, 3, 3), dtype=np.float64)
        for joint in range(77):
            for axis in range(3):
                resampled[:, joint, axis] = np.interp(
                    new_seconds, old_seconds, positions[:, joint, axis]
                )
            new_rotations[:, joint] = Slerp(old_seconds, Rotation.from_matrix(rotations[:, joint]))(
                new_seconds
            ).as_matrix()
        positions, rotations = resampled, new_rotations
    floor_y = 0.0
    if ground:
        toes = [
            SOMA77_JOINT_NAMES.index(n)
            for n in ("LeftToeBase", "LeftToeEnd", "RightToeBase", "RightToeEnd")
        ]
        floor_y = float(np.quantile(positions[:, toes, 1], 0.01))
        positions[:, :, 1] -= floor_y
    metadata = {
        "converter": "bones-seed-bvh-to-soma77-v1",
        "source_path": str(source_path),
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "source_frames": len(bvh.values),
        "source_fps": 1.0 / bvh.frame_time,
        "output_fps": fps,
        "position_channels": "absolute_local_replace_offset",
        "position_scale": 0.01,
        "coordinate_system": "Y-up meters; loader applies Rx(+90deg)",
        "rotation_conversion": "global_bvh @ kimodo_global_rot_offsets.T",
        "rotation_offsets_sha256": sha256(
            (root_path() / "soma/soma77_bvh_rotation_offsets.npz").read_bytes()
        ).hexdigest(),
        "floor_y_subtracted": floor_y,
        "contact_labels": "absent; downstream geometric inference",
    }
    contact_arrays: dict[str, Any] = {}
    if estimate_contacts:
        contacts, provenance = estimate_bones_seed_contacts(positions, fps)
        provenance["source_bvh_sha256"] = metadata["source_sha256"]
        contact_arrays = {
            "foot_contacts": contacts,
            "foot_contact_source": np.asarray("bones_seed_kinematic_foot_point_contacts_v1"),
            "foot_contact_provenance": np.asarray(json.dumps(provenance, sort_keys=True)),
            "foot_contact_names": np.asarray(provenance["channel_names"]),
        }
        metadata["contact_labels"] = "estimated source kinematics; not measured contacts"
        metadata["contact_estimation"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, validated archive; an interrupted/full-disk write
    # must not leave an apparently usable output. Hard-link creation is exclusive.
    descriptor, temporary_name = tempfile.mkstemp(suffix=".npz", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                posed_joints=positions,
                global_rot_mats=rotations,
                root_positions=positions[:, 0],
                fps=np.asarray(fps),
                joint_names=np.asarray(SOMA77_JOINT_NAMES),
                conversion_metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **contact_arrays,
            )
        validate_soma_npz(temporary)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise MotionValidationError(f"Refusing to overwrite {output}.") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output
