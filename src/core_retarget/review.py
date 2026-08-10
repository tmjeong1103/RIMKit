"""Generate explicitly unreviewed artifacts at the Stage 3 boundary.

This module stops at initial collision (Stage 3).  It deliberately does not
label the resulting trajectory as final robot motion; the complete pipeline is
available separately through :func:`core_retarget.pipeline.run_retarget_pipeline`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core_retarget._version import __version__
from core_retarget.api import Retargeter
from core_retarget.assets import root_path
from core_retarget.config.schema import RunConfig
from core_retarget.exceptions import ArtifactError
from core_retarget.motion.soma import load_soma_motion
from core_retarget.native import BackendPreference
from core_retarget.render import (
    LEGACY_HEIGHT,
    LEGACY_WIDTH,
    PreviewArtifacts,
    build_preview_contact_state,
    render_motion_preview,
)
from core_retarget.stages import DmrProgress, InitialCollisionProgress

_ARTIFACT_SCHEMA_VERSION = 2
_CONTACT_FORMAT = "core-preview-contacts-v1"
_DMR_FORMAT = "core-dmr-review-v1"
_INITIAL_COLLISION_FORMAT = "core-initial-collision-review-v1"


@dataclass(frozen=True, slots=True)
class ReviewRunResult:
    """Paths published by one successful Stage 3 review run."""

    output_dir: Path
    manifest_path: Path
    contacts_path: Path
    dmr_path: Path
    initial_collision_path: Path
    video_path: Path | None
    thumbnail_path: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _source_label(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _archive_metadata(
    *,
    artifact_format: str,
    robot_id: str,
    model_sha256: str,
    source_sha256: str,
) -> dict[str, NDArray[np.generic]]:
    return {
        "format": np.asarray(artifact_format),
        "robot_id": np.asarray(robot_id),
        "model_sha256": np.asarray(model_sha256),
        "source_motion_sha256": np.asarray(source_sha256),
    }


def _write_npz(
    path: Path,
    metadata: Mapping[str, NDArray[np.generic]],
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    payload = dict(metadata)
    overlap = set(payload).intersection(arrays)
    if overlap:
        raise ArtifactError(f"Artifact metadata collides with stage arrays: {sorted(overlap)}")
    payload.update(arrays)
    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ArtifactError(f"Object arrays are forbidden in review artifacts: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ArtifactError(f"Non-finite values are forbidden in review artifacts: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy's stub reserves one named keyword while the runtime intentionally
    # accepts arbitrary array names for NPZ members.
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactError("Review manifest is not strict JSON.") from error


def _preview_record(preview: PreviewArtifacts, root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "width": preview.width,
        "height": preview.height,
        "fps": preview.fps,
        "frame_count": preview.frame_count,
        "camera": asdict(preview.camera),
        "style": preview.visualization_style,
        "contact_overlay": preview.contact_overlay,
        "trajectory_stage": "initial_collision",
    }
    if preview.video_path is not None:
        payload["video"] = _artifact_record(preview.video_path, root)
    if preview.thumbnail_path is not None:
        payload["thumbnail"] = _artifact_record(preview.thumbnail_path, root)
    return payload


def run_review(
    motion_path: str | Path,
    robot_id: str,
    output_dir: str | Path,
    *,
    render_video: bool = False,
    render_thumbnail: bool = False,
    width: int = LEGACY_WIDTH,
    height: int = LEGACY_HEIGHT,
    fps_override: float | None = None,
    dmr_progress: DmrProgress | None = None,
    collision_progress: InitialCollisionProgress | None = None,
    backend: BackendPreference = "auto",
) -> ReviewRunResult:
    """Run DMR and Stage 3, then atomically publish review artifacts.

    ``output_dir`` must not already exist.  This prevents an interrupted or
    repeated run from silently mixing artifacts produced by different code or
    environments.
    """

    source = Path(motion_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ArtifactError(
            f"Review output already exists: {destination}. Choose a new directory "
            "or remove the old review explicitly."
        )
    if not source.is_file():
        raise ArtifactError(f"Source motion does not exist: {source}")
    if destination.parent.exists() and not destination.parent.is_dir():
        raise ArtifactError(f"Review output parent is not a directory: {destination.parent}")

    retargeter = Retargeter(
        robot_id,
        RunConfig(robot=robot_id, fps=fps_override, backend=backend),
    )
    preflight = retargeter.preflight(source)
    motion = load_soma_motion(source, fps_override=fps_override)
    source_sha256 = preflight.motion.sha256
    spec = retargeter.robot
    asset_root = root_path().resolve()
    scene_path = (asset_root / spec.scene_relpath).resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    published = False
    try:
        dmr = retargeter.run_dmr(source, progress=dmr_progress)
        collision = retargeter.run_initial_collision(dmr, progress=collision_progress)
        contacts = build_preview_contact_state(
            motion,
            qpos=collision.qpos,
            robot_id=spec.robot_id,
        )

        contacts_path = temporary / "stages" / "1_contacts.npz"
        dmr_path = temporary / "stages" / "2_dmr.npz"
        collision_path = temporary / "stages" / "3_initial_collision.npz"
        contact_metadata = _archive_metadata(
            artifact_format=_CONTACT_FORMAT,
            robot_id=spec.robot_id,
            model_sha256=spec.model_sha256,
            source_sha256=source_sha256,
        )
        _write_npz(contacts_path, contact_metadata, contacts.reference_arrays())
        metadata = _archive_metadata(
            artifact_format=_DMR_FORMAT,
            robot_id=spec.robot_id,
            model_sha256=spec.model_sha256,
            source_sha256=source_sha256,
        )
        _write_npz(dmr_path, metadata, dmr.reference_arrays())
        collision_metadata = _archive_metadata(
            artifact_format=_INITIAL_COLLISION_FORMAT,
            robot_id=spec.robot_id,
            model_sha256=spec.model_sha256,
            source_sha256=source_sha256,
        )
        _write_npz(collision_path, collision_metadata, collision.reference_arrays())

        video_path = temporary / "preview" / "stage3.mp4" if render_video else None
        thumbnail_path = temporary / "preview" / "stage3.png" if render_thumbnail else None
        preview: PreviewArtifacts | None = None
        if video_path is not None or thumbnail_path is not None:
            preview = render_motion_preview(
                robot_id=spec.robot_id,
                qpos=collision.qpos,
                fps=collision.fps,
                motion_name=source.stem,
                contacts=contacts,
                video_path=video_path,
                thumbnail_path=thumbnail_path,
                width=width,
                height=height,
            )

        artifacts: dict[str, Any] = {
            "contacts": _artifact_record(contacts_path, temporary),
            "dmr": _artifact_record(dmr_path, temporary),
            "initial_collision": _artifact_record(collision_path, temporary),
        }
        if preview is not None:
            artifacts["preview"] = _preview_record(preview, temporary)

        manifest: dict[str, Any] = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "classification": "stage3-review",
            "review_status": "unreviewed",
            "pipeline_complete": False,
            "last_completed_stage": "initial_collision",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "core_version": __version__,
            "compute_backend": retargeter.backend.manifest_record(),
            "source_motion": {
                "path": _source_label(source),
                "sha256": source_sha256,
                "frame_count": preflight.motion.frame_count,
                "fps": preflight.motion.fps,
                "duration_seconds": preflight.motion.duration_seconds,
            },
            "robot": {
                "id": spec.robot_id,
                "manufacturer": spec.manufacturer,
                "model": spec.display_name,
                "model_sha256": spec.model_sha256,
                "scene_sha256": _sha256(scene_path),
            },
            "environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": _dependency_version("scipy"),
                "mujoco": _dependency_version("mujoco"),
                "cvxpy": _dependency_version("cvxpy"),
                "clarabel": _dependency_version("clarabel"),
            },
            "diagnostics": asdict(collision.diagnostics),
            "artifacts": artifacts,
            "warnings": [
                "This artifact stops at Stage 3 and is not final CoRe output.",
                "Stage 3 does not apply foot grounding; foot-floor penetration may be present.",
                "Visual and physical motion quality has not been approved.",
            ]
            + (
                [
                    "The native accelerator was unavailable, so this run used the portable "
                    "Python reference backend."
                ]
                if retargeter.backend.reason == "native_extension_unavailable"
                else []
            ),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_strict_json_bytes(manifest))

        if destination.exists():
            raise ArtifactError(f"Review output appeared during the run: {destination}")
        temporary.chmod(0o755)
        temporary.replace(destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)

    return ReviewRunResult(
        output_dir=destination,
        manifest_path=destination / "manifest.json",
        contacts_path=destination / "stages" / "1_contacts.npz",
        dmr_path=destination / "stages" / "2_dmr.npz",
        initial_collision_path=destination / "stages" / "3_initial_collision.npz",
        video_path=destination / "preview" / "stage3.mp4" if render_video else None,
        thumbnail_path=(destination / "preview" / "stage3.png" if render_thumbnail else None),
    )


__all__ = ["ReviewRunResult", "run_review"]
