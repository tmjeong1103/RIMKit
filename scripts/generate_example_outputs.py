#!/usr/bin/env python3
"""Generate complete CoRe outputs for a bundled Kimodo or GEM-X motion set.

The default matrix contains all eight bundled Kimodo motions and all supported
robots. Select ``--source-set gem-x`` for the eight bundled GEM-X motions.
Complete run bundles are written below ``runs/example-outputs``. Use
``--gallery-dir`` (optionally without a value) to publish portable final MP4
and PNG files below ``docs/media/final``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, cast

import numpy as np

SCRIPT_PATH: Final = Path(__file__).resolve()
REPOSITORY_ROOT: Final = SCRIPT_PATH.parents[1]
KIMODO_MOTION_ROOT: Final = REPOSITORY_ROOT / "examples/motions/kimodo/soma_rp_v11"
GEMX_MOTION_ROOT: Final = REPOSITORY_ROOT / "examples/motions/gem-x"
DEFAULT_OUTPUT: Final = Path("runs/example-outputs")
DEFAULT_GALLERY: Final = Path("docs/media/final")
RESULT_RECORD_NAME: Final = "batch-result.json"
GALLERY_INDEX_NAME: Final = "index.json"
BATCH_SCHEMA_VERSION: Final = 1
PIPELINE_MANIFEST_SCHEMA_VERSION: Final = 2
PIPELINE_CLASSIFICATION: Final = "core-final-candidate"
GALLERY_CLASSIFICATION: Final = "core-final-candidate-gallery"
VISUALIZATION_STYLE: Final = "legacy-contact-overlay-v1"

KIMODO_MOTIONS: Final[dict[str, Path]] = {
    "alternating_lunges_contacts": KIMODO_MOTION_ROOT / "alternating_lunges_contacts.npz",
    "backward_walk_contacts": KIMODO_MOTION_ROOT / "backward_walk_contacts.npz",
    "foot_walk_stop": KIMODO_MOTION_ROOT / "foot_walk_stop.npz",
    "jump_land_contacts": KIMODO_MOTION_ROOT / "jump_land_contacts.npz",
    "side_steps_right_contacts": KIMODO_MOTION_ROOT / "side_steps_right_contacts.npz",
    "slow_walk_firm_steps": KIMODO_MOTION_ROOT / "slow_walk_firm_steps.npz",
    "stand_walk_run_stop": KIMODO_MOTION_ROOT / "stand_walk_run_stop.npz",
    "march_in_place_contacts": KIMODO_MOTION_ROOT / "march_in_place_contacts.npz",
}
GEMX_MOTIONS: Final[dict[str, Path]] = {
    "rapid_stepping": GEMX_MOTION_ROOT / "rapid_stepping.pt",
    "leg_stretching": GEMX_MOTION_ROOT / "leg_stretching.pt",
    "scurry_walk": GEMX_MOTION_ROOT / "scurry_walk.pt",
    "scurry_walk2": GEMX_MOTION_ROOT / "scurry_walk2.pt",
    "side_step": GEMX_MOTION_ROOT / "side_step.pt",
    "small_steps": GEMX_MOTION_ROOT / "small_steps.pt",
    "small_steps2": GEMX_MOTION_ROOT / "small_steps2.pt",
    "walk_with_short_stride": GEMX_MOTION_ROOT / "walk_with_short_stride.pt",
}
MOTION_SETS: Final[dict[str, dict[str, Path]]] = {
    "kimodo": KIMODO_MOTIONS,
    "gem-x": GEMX_MOTIONS,
}
SOURCE_FPS: Final[dict[str, float | None]] = {"kimodo": None, "gem-x": 30.0}
ALL_MOTIONS: Final[dict[str, Path]] = {**KIMODO_MOTIONS, **GEMX_MOTIONS}
ROBOTS: Final = (
    "g1",
    "h1",
    "h2",
    "r1",
    "k1",
    "apollo",
    "oli",
    "n1",
    "adam",
    "t1",
    "pm01",
)
REQUIRED_STAGES: Final = (
    "1_contacts",
    "2_dmr",
    "3_initial_collision",
    "4_target_trajectories",
    "5_ara",
    "6_fpa_targets",
    "7_fpa_ik",
    "8_final",
    "9_diagnostics",
)


class BatchGenerationError(RuntimeError):
    """Raised when a batch input, result, or artifact is unsafe or incomplete."""


class PipelineResultLike(Protocol):
    """Return contract consumed from the public end-to-end pipeline."""

    output_dir: Path
    manifest_path: Path
    final_motion_path: Path
    stage_paths: Mapping[str, Path]
    video_path: Path | None
    thumbnail_path: Path | None


class RunPipelineLike(Protocol):
    """Callable contract for ``core_retarget.pipeline.run_retarget_pipeline``."""

    def __call__(
        self,
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        *,
        fps_override: float | None,
        save_stages: bool,
        render_video: bool,
        render_thumbnail: bool,
        width: int,
        height: int,
    ) -> PipelineResultLike: ...


@dataclass(frozen=True, slots=True)
class ValidatedOutput:
    """Hash-verified files belonging to one complete pipeline output."""

    motion_id: str
    robot_id: str
    output_dir: Path
    manifest_path: Path
    final_motion_path: Path
    stage_paths: Mapping[str, Path]
    video_path: Path
    thumbnail_path: Path


@dataclass(frozen=True, slots=True)
class GalleryRenderInput:
    """Hash-verified final motion needed to refresh portable gallery media."""

    motion_id: str
    robot_id: str
    final_motion_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments without importing the MuJoCo pipeline runtime."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-set",
        choices=tuple(MOTION_SETS),
        default="kimodo",
        help="bundled source-motion set to run",
    )
    parser.add_argument(
        "--motion",
        action="append",
        choices=tuple(ALL_MOTIONS),
        help="bundled motion to run; repeat to select multiple motions",
    )
    parser.add_argument(
        "--robot",
        action="append",
        choices=ROBOTS,
        help="robot to run; repeat to select multiple robots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="root for complete per-motion, per-robot pipeline bundles",
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_GALLERY,
        help="optionally publish <motion>/<robot>.mp4|png and index.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip combinations with a complete, hash-verified batch-result.json",
    )
    parser.add_argument(
        "--rerender-gallery",
        action="store_true",
        help=(
            "reuse complete, hash-verified robot motions and refresh only gallery "
            "MP4/PNG media; defaults --gallery-dir to docs/media/final"
        ),
    )
    return parser.parse_args(argv)


def _deduplicate(values: Sequence[str] | None, defaults: Sequence[str]) -> tuple[str, ...]:
    selected = defaults if values is None else values
    return tuple(dict.fromkeys(selected))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchGenerationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BatchGenerationError(f"{label.capitalize()} must contain a JSON object: {path}")
    return payload


def _portable_relative_path(root: Path, path: Path, *, field: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BatchGenerationError(f"{field} is outside its pipeline output: {path}") from exc
    portable = PurePosixPath(relative.as_posix())
    if portable.is_absolute() or ".." in portable.parts or portable.as_posix() == ".":
        raise BatchGenerationError(f"{field} is not a safe artifact path: {path}")
    return portable.as_posix()


def _resolve_recorded_artifact(output_dir: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BatchGenerationError(f"{field} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BatchGenerationError(f"{field} escapes its pipeline output: {value!r}")
    path = output_dir.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise BatchGenerationError(f"{field} escapes its pipeline output: {value!r}") from exc
    return path


def _validate_file(path: Path, *, suffix: str, field: str) -> Path:
    if path.suffix.lower() != suffix:
        raise BatchGenerationError(f"{field} must be a {suffix} file: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise BatchGenerationError(f"{field} is missing or empty: {path}")
    return path


def _validate_npz(path: Path, *, field: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise BatchGenerationError(f"{field} contains no arrays: {path}")
            for name in archive.files:
                array = archive[name]
                if array.dtype.hasobject:
                    raise BatchGenerationError(
                        f"{field} contains a forbidden object array {name!r}: {path}"
                    )
    except BatchGenerationError:
        raise
    except (OSError, ValueError) as exc:
        raise BatchGenerationError(f"Could not validate {field} {path}: {exc}") from exc


def _validate_artifact_descriptor(
    output_dir: Path,
    descriptor: Mapping[str, Any],
    *,
    field: str,
    suffix: str,
    validate_npz: bool = False,
) -> Path:
    path = _resolve_recorded_artifact(
        output_dir,
        descriptor.get("path"),
        field=f"{field}.path",
    )
    _validate_file(path, suffix=suffix, field=field)
    expected_size = descriptor.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise BatchGenerationError(f"{field}.size_bytes is invalid")
    if path.stat().st_size != expected_size:
        raise BatchGenerationError(f"Pipeline artifact size mismatch for {path}")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(expected_sha256, str) or _sha256(path) != expected_sha256:
        raise BatchGenerationError(f"Pipeline artifact SHA-256 mismatch for {path}")
    if validate_npz:
        _validate_npz(path, field=field)
    return path


def _descriptor(path: Path, output_dir: Path, *, field: str) -> dict[str, Any]:
    return {
        "path": _portable_relative_path(output_dir, path, field=field),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_pipeline_manifest(
    path: Path,
    *,
    motion_id: str,
    motion_path: Path,
    robot_id: str,
) -> ValidatedOutput:
    _validate_file(path, suffix=".json", field="pipeline manifest")
    payload = _load_json_object(path, label="pipeline manifest")
    if payload.get("schema_version") != PIPELINE_MANIFEST_SCHEMA_VERSION:
        raise BatchGenerationError(f"Unsupported pipeline manifest schema in {path}")
    if payload.get("classification") != PIPELINE_CLASSIFICATION:
        raise BatchGenerationError(f"Unexpected pipeline classification in {path}")
    if payload.get("review_status") != "unreviewed" or payload.get("pipeline_complete") is not True:
        raise BatchGenerationError(f"Pipeline completion contract is invalid in {path}")

    robot = payload.get("robot")
    if not isinstance(robot, dict) or robot.get("id") != robot_id:
        raise BatchGenerationError(f"Pipeline robot identity mismatch in {path}")
    source = payload.get("source_motion")
    if not isinstance(source, dict) or source.get("sha256") != _sha256(motion_path):
        raise BatchGenerationError(f"Pipeline source-motion identity mismatch in {path}")

    output_dir = path.parent.resolve()
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise BatchGenerationError(f"Pipeline artifacts must be an object in {path}")
    final_descriptor = artifacts.get("final_motion")
    if not isinstance(final_descriptor, dict):
        raise BatchGenerationError(f"Pipeline final_motion must be an object in {path}")
    final_motion_path = _validate_artifact_descriptor(
        output_dir,
        final_descriptor,
        field="artifacts.final_motion",
        suffix=".npz",
        validate_npz=True,
    )

    stages = artifacts.get("stages")
    if not isinstance(stages, dict):
        raise BatchGenerationError(f"Pipeline stages must be an object in {path}")
    if set(stages) != set(REQUIRED_STAGES):
        missing = sorted(set(REQUIRED_STAGES).difference(stages))
        unexpected = sorted(set(stages).difference(REQUIRED_STAGES))
        raise BatchGenerationError(
            f"Pipeline stage set is incomplete in {path}; missing={missing}, "
            f"unexpected={unexpected}"
        )
    stage_paths: dict[str, Path] = {}
    for stage_name in REQUIRED_STAGES:
        stage_descriptor = stages.get(stage_name)
        if not isinstance(stage_descriptor, dict):
            raise BatchGenerationError(f"Pipeline stage {stage_name} must be an object in {path}")
        stage_paths[stage_name] = _validate_artifact_descriptor(
            output_dir,
            stage_descriptor,
            field=f"artifacts.stages.{stage_name}",
            suffix=".npz",
            validate_npz=True,
        )

    preview = artifacts.get("preview")
    if not isinstance(preview, dict):
        raise BatchGenerationError(f"Pipeline preview must be an object in {path}")
    if (
        preview.get("style") != VISUALIZATION_STYLE
        or preview.get("contact_overlay") is not True
        or preview.get("trajectory_stage") != "final"
        or preview.get("width") != 1280
        or preview.get("height") != 720
    ):
        raise BatchGenerationError(f"Pipeline visualization contract is invalid in {path}")
    media: dict[str, Path] = {}
    for name, suffix in (("video", ".mp4"), ("thumbnail", ".png")):
        media_descriptor = preview.get(name)
        if not isinstance(media_descriptor, dict):
            raise BatchGenerationError(f"Pipeline preview {name} must be an object in {path}")
        media[name] = _validate_artifact_descriptor(
            output_dir,
            media_descriptor,
            field=f"artifacts.preview.{name}",
            suffix=suffix,
        )

    return ValidatedOutput(
        motion_id=motion_id,
        robot_id=robot_id,
        output_dir=output_dir,
        manifest_path=path.resolve(),
        final_motion_path=final_motion_path,
        stage_paths=stage_paths,
        video_path=media["video"],
        thumbnail_path=media["thumbnail"],
    )


def _validate_gallery_render_input(
    output_dir: Path,
    *,
    motion_id: str,
    motion_path: Path,
    robot_id: str,
) -> GalleryRenderInput:
    """Validate the immutable identity and final NPZ needed for rerendering.

    Schema 1 completed bundles remain valid gallery inputs. Schema 2 adds
    compute-backend provenance but does not change the final-motion contract.
    """

    manifest_path = output_dir / "manifest.json"
    _validate_file(manifest_path, suffix=".json", field="pipeline manifest")
    payload = _load_json_object(manifest_path, label="pipeline manifest")
    if payload.get("schema_version") not in {1, PIPELINE_MANIFEST_SCHEMA_VERSION}:
        raise BatchGenerationError(f"Unsupported pipeline manifest schema in {manifest_path}")
    if (
        payload.get("classification") != PIPELINE_CLASSIFICATION
        or payload.get("review_status") != "unreviewed"
        or payload.get("pipeline_complete") is not True
    ):
        raise BatchGenerationError(f"Pipeline completion contract is invalid in {manifest_path}")
    robot = payload.get("robot")
    if not isinstance(robot, dict) or robot.get("id") != robot_id:
        raise BatchGenerationError(f"Pipeline robot identity mismatch in {manifest_path}")
    source = payload.get("source_motion")
    if not isinstance(source, dict) or source.get("sha256") != _sha256(motion_path):
        raise BatchGenerationError(f"Pipeline source-motion identity mismatch in {manifest_path}")
    artifacts = payload.get("artifacts")
    final_descriptor = artifacts.get("final_motion") if isinstance(artifacts, dict) else None
    if not isinstance(final_descriptor, dict):
        raise BatchGenerationError(f"Pipeline final_motion must be an object in {manifest_path}")
    final_motion_path = _validate_artifact_descriptor(
        output_dir,
        final_descriptor,
        field="artifacts.final_motion",
        suffix=".npz",
        validate_npz=True,
    )
    return GalleryRenderInput(
        motion_id=motion_id,
        robot_id=robot_id,
        final_motion_path=final_motion_path,
    )


def _result_path(
    value: Path | None,
    output_dir: Path,
    *,
    field: str,
    suffix: str,
) -> Path:
    if value is None:
        raise BatchGenerationError(f"Pipeline did not return {field}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    path = path.resolve()
    _portable_relative_path(output_dir, path, field=field)
    return _validate_file(path, suffix=suffix, field=field)


def _record_completed(output: ValidatedOutput) -> None:
    payload = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "classification": "core-final-candidate-batch-result",
        "motion_id": output.motion_id,
        "robot_id": output.robot_id,
        "manifest": _descriptor(
            output.manifest_path,
            output.output_dir,
            field="manifest",
        ),
        "final_motion": _descriptor(
            output.final_motion_path,
            output.output_dir,
            field="final_motion",
        ),
        "stages": {
            name: _descriptor(path, output.output_dir, field=f"stages.{name}")
            for name, path in output.stage_paths.items()
        },
        "video": _descriptor(output.video_path, output.output_dir, field="video"),
        "thumbnail": _descriptor(
            output.thumbnail_path,
            output.output_dir,
            field="thumbnail",
        ),
    }
    _write_json_atomic(output.output_dir / RESULT_RECORD_NAME, payload)


def _validate_record_descriptor(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    name: str,
    suffix: str,
    validate_npz: bool = False,
) -> Path:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise BatchGenerationError(f"Resume record {name} must be an object")
    return _validate_artifact_descriptor(
        output_dir,
        value,
        field=f"resume.{name}",
        suffix=suffix,
        validate_npz=validate_npz,
    )


def _resume_completed(
    output_dir: Path,
    *,
    motion_id: str,
    motion_path: Path,
    robot_id: str,
) -> ValidatedOutput | None:
    record_path = output_dir / RESULT_RECORD_NAME
    if not record_path.is_file():
        if output_dir.exists():
            raise BatchGenerationError(
                f"Cannot safely resume {motion_id}/{robot_id}: {output_dir} exists but "
                f"has no {RESULT_RECORD_NAME}. Move or remove that incomplete directory first."
            )
        return None

    payload = _load_json_object(record_path, label="resume record")
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise BatchGenerationError(f"Unsupported resume record schema: {record_path}")
    if payload.get("motion_id") != motion_id or payload.get("robot_id") != robot_id:
        raise BatchGenerationError(f"Resume record identity mismatch: {record_path}")

    manifest_path = _validate_record_descriptor(
        output_dir,
        payload,
        name="manifest",
        suffix=".json",
    )
    validated = _validate_pipeline_manifest(
        manifest_path,
        motion_id=motion_id,
        motion_path=motion_path,
        robot_id=robot_id,
    )

    recorded_final = _validate_record_descriptor(
        output_dir,
        payload,
        name="final_motion",
        suffix=".npz",
        validate_npz=True,
    )
    if recorded_final != validated.final_motion_path:
        raise BatchGenerationError(f"Resume final-motion path mismatch: {record_path}")

    recorded_stages = payload.get("stages")
    if not isinstance(recorded_stages, dict) or set(recorded_stages) != set(REQUIRED_STAGES):
        raise BatchGenerationError(f"Resume record stage set is invalid: {record_path}")
    for stage_name, expected_path in validated.stage_paths.items():
        recorded_path = _validate_record_descriptor(
            output_dir,
            recorded_stages,
            name=stage_name,
            suffix=".npz",
            validate_npz=True,
        )
        if recorded_path != expected_path:
            raise BatchGenerationError(
                f"Resume stage path mismatch for {stage_name}: {record_path}"
            )

    for name, suffix, expected_path in (
        ("video", ".mp4", validated.video_path),
        ("thumbnail", ".png", validated.thumbnail_path),
    ):
        recorded_path = _validate_record_descriptor(
            output_dir,
            payload,
            name=name,
            suffix=suffix,
        )
        if recorded_path != expected_path:
            raise BatchGenerationError(f"Resume {name} path mismatch: {record_path}")
    return validated


def _load_pipeline_runner() -> RunPipelineLike:
    try:
        from core_retarget.pipeline import run_retarget_pipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise BatchGenerationError(
            "core_retarget.pipeline.run_retarget_pipeline is unavailable. "
            "Install this CoRe checkout with the complete pipeline implementation."
        ) from exc
    return cast(RunPipelineLike, run_retarget_pipeline)


def _run_one(
    run_pipeline: RunPipelineLike,
    *,
    motion_id: str,
    motion_path: Path,
    fps_override: float | None,
    robot_id: str,
    output_dir: Path,
) -> ValidatedOutput:
    result = run_pipeline(
        motion_path,
        robot_id,
        output_dir,
        fps_override=fps_override,
        save_stages=True,
        render_video=True,
        render_thumbnail=True,
        width=1280,
        height=720,
    )
    result_output = Path(result.output_dir).expanduser()
    if not result_output.is_absolute():
        result_output = output_dir / result_output
    result_output = result_output.resolve()
    if result_output != output_dir.resolve():
        raise BatchGenerationError(
            "Pipeline returned a different output directory: "
            f"expected {output_dir}, found {result_output}"
        )

    manifest_path = _result_path(
        result.manifest_path,
        result_output,
        field="manifest_path",
        suffix=".json",
    )
    validated = _validate_pipeline_manifest(
        manifest_path,
        motion_id=motion_id,
        motion_path=motion_path,
        robot_id=robot_id,
    )
    result_final = _result_path(
        result.final_motion_path,
        result_output,
        field="final_motion_path",
        suffix=".npz",
    )
    if result_final != validated.final_motion_path:
        raise BatchGenerationError("Pipeline result and manifest disagree on final_motion_path")
    if set(result.stage_paths) != set(REQUIRED_STAGES):
        raise BatchGenerationError("Pipeline result returned an incomplete stage-path mapping")
    for name, expected_path in validated.stage_paths.items():
        result_stage = _result_path(
            result.stage_paths[name],
            result_output,
            field=f"stage_paths.{name}",
            suffix=".npz",
        )
        if result_stage != expected_path:
            raise BatchGenerationError(f"Pipeline result and manifest disagree on stage {name}")
    result_video = _result_path(
        result.video_path,
        result_output,
        field="video_path",
        suffix=".mp4",
    )
    result_thumbnail = _result_path(
        result.thumbnail_path,
        result_output,
        field="thumbnail_path",
        suffix=".png",
    )
    if result_video != validated.video_path or result_thumbnail != validated.thumbnail_path:
        raise BatchGenerationError("Pipeline result and manifest disagree on preview media")
    _record_completed(validated)
    return validated


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _gallery_index(gallery_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for motion_id in ALL_MOTIONS:
        for robot_id in ROBOTS:
            for suffix, media_type in ((".mp4", "video"), (".png", "thumbnail")):
                path = gallery_dir / motion_id / f"{robot_id}{suffix}"
                if not path.is_file():
                    continue
                files.append(
                    {
                        "media_type": media_type,
                        "motion_id": motion_id,
                        "path": path.relative_to(gallery_dir).as_posix(),
                        "robot_id": robot_id,
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "classification": GALLERY_CLASSIFICATION,
        "review_status": "unreviewed",
        "pipeline_complete": True,
        "generator": "scripts/generate_example_outputs.py",
        "visualization": {
            "style": VISUALIZATION_STYLE,
            "width": 1280,
            "height": 720,
            "contact_overlay": True,
            "trajectory_stage": "final",
        },
        "files": files,
    }


def _publish_gallery(gallery_dir: Path, outputs: Sequence[ValidatedOutput]) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        motion_dir = gallery_dir / output.motion_id
        _copy_atomic(output.video_path, motion_dir / f"{output.robot_id}.mp4")
        _copy_atomic(output.thumbnail_path, motion_dir / f"{output.robot_id}.png")
    _write_json_atomic(gallery_dir / GALLERY_INDEX_NAME, _gallery_index(gallery_dir))


def _rerender_gallery_output(
    output: GalleryRenderInput,
    gallery_dir: Path,
    *,
    motion_path: Path,
    fps_override: float | None,
) -> None:
    """Render portable gallery media from a validated final-motion artifact."""

    from core_retarget.motion import load_source_motion
    from core_retarget.render import build_preview_contact_state, render_motion_preview

    try:
        with np.load(output.final_motion_path, allow_pickle=False) as archive:
            required = {"robot_id", "fps", "qpos"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise BatchGenerationError(
                    f"Final motion is missing required arrays {missing}: {output.final_motion_path}"
                )
            recorded_robot = str(np.asarray(archive["robot_id"]).item())
            fps = float(np.asarray(archive["fps"]).item())
            qpos = np.array(archive["qpos"], dtype=np.float64, copy=True, order="C")
    except BatchGenerationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise BatchGenerationError(
            f"Could not load final motion for gallery rendering: {output.final_motion_path}"
        ) from exc

    if recorded_robot != output.robot_id:
        raise BatchGenerationError(
            f"Final-motion robot mismatch: expected {output.robot_id}, found {recorded_robot}"
        )
    motion = load_source_motion(motion_path, fps_override=fps_override).motion
    if qpos.shape[0] != motion.frame_count or not np.isclose(fps, motion.fps):
        raise BatchGenerationError(
            f"Final motion and source timing disagree for {output.motion_id}/{output.robot_id}"
        )
    contacts = build_preview_contact_state(
        motion,
        qpos=qpos,
        robot_id=output.robot_id,
    )

    motion_dir = gallery_dir / output.motion_id
    motion_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=motion_dir,
        prefix=f".{output.robot_id}-rerender-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        video = temporary / f"{output.robot_id}.mp4"
        thumbnail = temporary / f"{output.robot_id}.png"
        render_motion_preview(
            robot_id=output.robot_id,
            qpos=qpos,
            fps=fps,
            motion_name=output.motion_id,
            contacts=contacts,
            video_path=video,
            thumbnail_path=thumbnail,
            width=1280,
            height=720,
            source_provider="gem-x" if motion_path.suffix.lower() == ".pt" else "kimodo",
        )
        _copy_atomic(video, motion_dir / video.name)
        _copy_atomic(thumbnail, motion_dir / thumbnail.name)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    motions = MOTION_SETS[args.source_set]
    invalid_motion_ids = sorted(set(args.motion or ()).difference(motions))
    if invalid_motion_ids:
        raise BatchGenerationError(
            f"Motion(s) {', '.join(invalid_motion_ids)} do not belong to "
            f"the {args.source_set} source set."
        )
    motion_ids = _deduplicate(args.motion, tuple(motions))
    fps_override = SOURCE_FPS[args.source_set]
    robot_ids = _deduplicate(args.robot, ROBOTS)
    output_root = args.output.expanduser().resolve()
    combinations = [(motion_id, robot_id) for motion_id in motion_ids for robot_id in robot_ids]

    for motion_path in (motions[motion_id] for motion_id in motion_ids):
        if not motion_path.is_file():
            raise BatchGenerationError(f"Bundled motion is missing: {motion_path}")

    if args.rerender_gallery:
        gallery_dir = (
            (DEFAULT_GALLERY if args.gallery_dir is None else args.gallery_dir)
            .expanduser()
            .resolve()
        )
        print(
            f"Rerendering {len(combinations)} gallery output(s) under {gallery_dir}",
            flush=True,
        )
        for index, (motion_id, robot_id) in enumerate(combinations, start=1):
            output_dir = output_root / motion_id / robot_id
            label = f"[{index}/{len(combinations)}] {motion_id}/{robot_id}"
            gallery_input = _validate_gallery_render_input(
                output_dir,
                motion_id=motion_id,
                motion_path=motions[motion_id],
                robot_id=robot_id,
            )
            print(f"{label} RENDER", flush=True)
            _rerender_gallery_output(
                gallery_input,
                gallery_dir,
                motion_path=motions[motion_id],
                fps_override=fps_override,
            )
            print(f"{label} DONE", flush=True)
        _write_json_atomic(gallery_dir / GALLERY_INDEX_NAME, _gallery_index(gallery_dir))
        print(
            f"Gallery refreshed: {len(combinations) * 2} media file(s), "
            f"index={gallery_dir / GALLERY_INDEX_NAME}",
            flush=True,
        )
        return 0

    if not args.resume:
        occupied = [
            output_root / motion_id / robot_id
            for motion_id, robot_id in combinations
            if (output_root / motion_id / robot_id).exists()
        ]
        if occupied:
            raise BatchGenerationError(
                "Refusing to overwrite an existing pipeline output. Use --resume for "
                "hash-verified completed runs or choose another --output: "
                f"{occupied[0]}"
            )

    run_pipeline = _load_pipeline_runner()
    print(
        f"Generating {len(combinations)} complete example output(s) under {output_root}",
        flush=True,
    )
    completed: list[ValidatedOutput] = []
    resumed = 0
    for index, (motion_id, robot_id) in enumerate(combinations, start=1):
        output_dir = output_root / motion_id / robot_id
        label = f"[{index}/{len(combinations)}] {motion_id}/{robot_id}"
        try:
            resumed_output = (
                _resume_completed(
                    output_dir,
                    motion_id=motion_id,
                    motion_path=motions[motion_id],
                    robot_id=robot_id,
                )
                if args.resume
                else None
            )
            if resumed_output is not None:
                completed.append(resumed_output)
                resumed += 1
                print(f"{label} RESUME verified complete result", flush=True)
                continue

            print(f"{label} RUN", flush=True)
            output = _run_one(
                run_pipeline,
                motion_id=motion_id,
                motion_path=motions[motion_id],
                fps_override=fps_override,
                robot_id=robot_id,
                output_dir=output_dir,
            )
            completed.append(output)
            print(
                f"{label} DONE final={output.final_motion_path} "
                f"video={output.video_path} thumbnail={output.thumbnail_path}",
                flush=True,
            )
        except Exception as exc:
            raise BatchGenerationError(f"{label} FAILED: {exc}") from exc

    if args.gallery_dir is not None:
        gallery_dir = args.gallery_dir.expanduser().resolve()
        _publish_gallery(gallery_dir, completed)
        print(
            f"Gallery: {len(completed) * 2} media file(s), "
            f"index={gallery_dir / GALLERY_INDEX_NAME}",
            flush=True,
        )

    print(
        f"Complete: {len(completed) - resumed} generated, {resumed} resumed, "
        f"{len(completed)} total",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchGenerationError as exc:
        print(f"generate_example_outputs: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
