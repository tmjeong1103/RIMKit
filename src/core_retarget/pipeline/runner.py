"""Shared end-to-end runner for the Python API, CLI, and future web UI."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from core_retarget._version import __version__
from core_retarget.assets import root_path
from core_retarget.exceptions import (
    ArtifactError,
    ModelVerificationError,
    MotionValidationError,
)
from core_retarget.export import write_robot_motion_npz
from core_retarget.motion import (
    LoadedSourceMotion,
    build_contact_schedule,
    load_soma_motion,
    load_source_motion,
)
from core_retarget.native import BackendPreference, BackendSelection, resolve_backend
from core_retarget.optimization.fpa import FpaSolveRecord
from core_retarget.pipeline.events import (
    EventSink,
    NullEventSink,
    PipelineEvent,
    PipelineEventType,
)
from core_retarget.pipeline.state import PipelineStage
from core_retarget.render import (
    LEGACY_HEIGHT,
    LEGACY_WIDTH,
    build_preview_contact_state,
    render_motion_preview,
)
from core_retarget.robots.registry import get_robot
from core_retarget.robots.validation import verify_robot
from core_retarget.stages import (
    run_ara,
    run_diagnostic_trajectories,
    run_dmr,
    run_final_collision,
    run_initial_collision,
    run_target_trajectories,
)
from core_retarget.stages.fpa import (
    FPA_IK_SOLVE_LABELS,
    FPA_TARGET_SOLVE_LABELS,
    AraResultLike,
    build_fpa_targets,
    solve_fpa,
)

_MANIFEST_SCHEMA_VERSION = 2
_CLASSIFICATION = "core-final-candidate"


@dataclass(frozen=True, slots=True)
class RetargetRunResult:
    """Paths atomically published by one completed end-to-end run."""

    robot_id: str
    output_dir: Path
    manifest_path: Path
    final_motion_path: Path
    stage_paths: Mapping[str, Path]
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


def _write_npz(
    path: Path,
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    payload = dict(arrays)
    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ArtifactError(f"Object arrays are forbidden in stage output: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ArtifactError(f"Non-finite values are forbidden in stage output: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != tuple(payload):
                raise ArtifactError(f"Stage archive members changed while writing {path}.")
            for name, expected in payload.items():
                actual = archive[name]
                expected_array = np.asarray(expected)
                if (
                    actual.dtype != expected_array.dtype
                    or actual.shape != expected_array.shape
                    or not np.array_equal(actual, expected_array)
                ):
                    raise ArtifactError(f"Stage archive member {name!r} changed on write.")
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Could not validate stage archive {path}: {exc}") from exc


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
    except (TypeError, ValueError) as exc:
        raise ArtifactError("Final manifest is not strict JSON.") from exc


def _ground_summary(
    distances: NDArray[np.float64],
    confidence: NDArray[np.float64],
) -> dict[str, float | int]:
    support = np.asarray(confidence, dtype=np.float64) >= 0.5
    values = np.asarray(distances, dtype=np.float64)[support]
    if len(values) == 0:
        return {
            "support_frames": 0,
            "min_m": 0.0,
            "median_m": 0.0,
            "p95_m": 0.0,
            "max_m": 0.0,
            "penetration_frames": 0,
        }
    return {
        "support_frames": int(len(values)),
        "min_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "max_m": float(np.max(values)),
        "penetration_frames": int(np.count_nonzero(values < -1e-9)),
    }


def _fpa_solver_diagnostics(
    target_records: Sequence[FpaSolveRecord],
    ik_records: Sequence[FpaSolveRecord],
    *,
    frame_count: int,
) -> dict[str, Any]:
    stage_60 = tuple(target_records)
    stage_70 = tuple(ik_records)
    expected_stage_70 = FPA_IK_SOLVE_LABELS if frame_count >= 4 else ()
    records_complete = (
        tuple(record.label for record in stage_60) == FPA_TARGET_SOLVE_LABELS
        and tuple(record.label for record in stage_70) == expected_stage_70
    )
    records = (*stage_60, *stage_70)
    all_reference_solver = all(record.solver == "CLARABEL" for record in records)
    all_optimal = all(record.status == "optimal" for record in records)
    return {
        "backend": "cvxpy",
        "reference_solver": "CLARABEL",
        "expected_counts": {
            "stage_60": len(FPA_TARGET_SOLVE_LABELS),
            "stage_70": len(expected_stage_70),
        },
        "actual_counts": {
            "stage_60": len(stage_60),
            "stage_70": len(stage_70),
        },
        "records_complete": records_complete,
        "all_reference_solver": all_reference_solver,
        "all_optimal": all_optimal,
        "solver_path_qualified": (records_complete and all_reference_solver and all_optimal),
        "stage_60": [asdict(record) for record in stage_60],
        "stage_70": [asdict(record) for record in stage_70],
    }


def _emit(
    sink: EventSink,
    *,
    event_type: PipelineEventType,
    stage: PipelineStage,
    job_id: str,
    robot_id: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    sink.emit(
        PipelineEvent(
            event_type=event_type,
            stage=stage,
            job_id=job_id,
            robot_id=robot_id,
            message=message,
            current=current,
            total=total,
            metrics={} if metrics is None else metrics,
        )
    )


def _emit_failure_safely(
    sink: EventSink,
    *,
    stage: PipelineStage,
    job_id: str,
    robot_id: str,
    error: Exception,
) -> None:
    """Report a terminal error without replacing the pipeline exception."""

    detail = str(error).strip()
    description = type(error).__name__ if not detail else f"{type(error).__name__}: {detail}"
    _emit_safely(
        sink,
        event_type=PipelineEventType.FAILED,
        stage=stage,
        job_id=job_id,
        robot_id=robot_id,
        message=f"{stage.value} failed ({description}).",
    )


def _emit_safely(
    sink: EventSink,
    *,
    event_type: PipelineEventType,
    stage: PipelineStage,
    job_id: str,
    robot_id: str,
    message: str,
    metrics: dict[str, float] | None = None,
) -> None:
    """Best-effort event delivery for failure and post-publication notices."""

    try:
        _emit(
            sink,
            event_type=event_type,
            stage=stage,
            job_id=job_id,
            robot_id=robot_id,
            message=message,
            metrics=metrics,
        )
    except Exception:
        # Event delivery is a side channel.  In these call sites, a broken sink
        # must neither mask the pipeline error nor invalidate published output.
        pass


def run_retarget_pipeline(
    motion_path: str | Path,
    robot_id: str,
    output_dir: str | Path,
    *,
    fps_override: float | None = None,
    save_stages: bool = True,
    render_video: bool = False,
    render_thumbnail: bool = False,
    width: int = LEGACY_WIDTH,
    height: int = LEGACY_HEIGHT,
    event_sink: EventSink | None = None,
    backend: BackendPreference | BackendSelection = "auto",
) -> RetargetRunResult:
    """Run every verified stage and atomically publish a self-describing bundle."""

    source = Path(motion_path).expanduser().resolve()
    source_provider: Literal["kimodo", "gem-x"] = (
        "kimodo" if source.suffix.lower() == ".npz" else "gem-x"
    )
    source_container = "npz" if source_provider == "kimodo" else "pt"
    destination = Path(output_dir).expanduser().resolve()
    robot = get_robot(robot_id)
    backend_selection = resolve_backend(backend)
    sink = NullEventSink() if event_sink is None else event_sink
    job_id = uuid.uuid4().hex
    active_stage = PipelineStage.VALIDATING
    temporary: Path | None = None
    published = False
    stage_paths: dict[str, Path] = {}
    video_path: Path | None = None
    thumbnail_path: Path | None = None
    final_motion_path: Path | None = None

    def stage_started(stage: PipelineStage, message: str) -> None:
        nonlocal active_stage
        active_stage = stage
        _emit(
            sink,
            event_type=PipelineEventType.STAGE_STARTED,
            stage=stage,
            job_id=job_id,
            robot_id=robot.robot_id,
            message=message,
        )

    def stage_completed(
        stage: PipelineStage,
        message: str,
        *,
        metrics: dict[str, float] | None = None,
    ) -> None:
        _emit(
            sink,
            event_type=PipelineEventType.STAGE_COMPLETED,
            stage=stage,
            job_id=job_id,
            robot_id=robot.robot_id,
            message=message,
            metrics=metrics,
        )

    try:
        stage_started(
            PipelineStage.VALIDATING,
            "Validating SOMA source input and robot model.",
        )
        if backend_selection.reason == "native_extension_unavailable":
            _emit(
                sink,
                event_type=PipelineEventType.WARNING,
                stage=PipelineStage.VALIDATING,
                job_id=job_id,
                robot_id=robot.robot_id,
                message=(
                    "The native accelerator is unavailable; this run is using the "
                    "portable Python reference backend."
                ),
            )
        if not source.is_file():
            raise ArtifactError(f"Source motion does not exist: {source}")
        if destination.exists():
            raise ArtifactError(
                f"Retarget output already exists: {destination}. Choose a new directory "
                "or remove the old output explicitly."
            )
        if destination.parent.exists() and not destination.parent.is_dir():
            raise ArtifactError(f"Retarget output parent is not a directory: {destination.parent}")

        loaded_source: LoadedSourceMotion | None
        if source_provider == "kimodo":
            motion = load_soma_motion(source, fps_override=fps_override)
            loaded_source = None
        else:
            loaded_source = load_source_motion(source, fps_override=fps_override)
            motion = loaded_source.motion
        if motion.frame_count < 2:
            raise MotionValidationError(
                "The complete CoRe pipeline requires at least two motion frames."
            )
        contacts = (
            build_contact_schedule(motion)
            if loaded_source is None
            else loaded_source.build_contact_schedule()
        )
        verification = verify_robot(robot, load_mujoco=True)
        if not verification.ok:
            details = "; ".join(issue.message for issue in verification.issues)
            raise ModelVerificationError(details)
        stage_completed(
            PipelineStage.VALIDATING,
            "SOMA input, contact schedule, and robot model are valid.",
            metrics={
                "frame_count": float(motion.frame_count),
                "fps": float(motion.fps),
            },
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.building-",
                dir=destination.parent,
            )
        )

        def dmr_progress(current: int, total: int, error: float) -> None:
            _emit(
                sink,
                event_type=PipelineEventType.PROGRESS,
                stage=PipelineStage.DMR,
                job_id=job_id,
                robot_id=robot.robot_id,
                message="Direct motion retargeting.",
                current=current,
                total=total,
                metrics={"ik_error": float(error)},
            )

        stage_started(PipelineStage.DMR, "Running direct motion retargeting.")
        dmr = run_dmr(
            motion,
            robot_id=robot.robot_id,
            progress=dmr_progress,
            backend=backend_selection,
            source_provider=source_provider,
            left_contact_confidence=contacts.left_confidence,
            right_contact_confidence=contacts.right_confidence,
        )
        stage_completed(
            PipelineStage.DMR,
            "Direct motion retargeting completed.",
            metrics={"frame_count": float(len(dmr.seconds))},
        )

        def collision_progress(
            outer_pass: int,
            current: int,
            total: int,
            margin: float,
        ) -> None:
            _emit(
                sink,
                event_type=PipelineEventType.PROGRESS,
                stage=PipelineStage.INITIAL_COLLISION,
                job_id=job_id,
                robot_id=robot.robot_id,
                message=f"Initial collision pass {outer_pass}.",
                current=current,
                total=total,
                metrics={"margin_m": float(margin)},
            )

        stage_started(
            PipelineStage.INITIAL_COLLISION,
            "Running initial arm self-collision refinement.",
        )
        initial_collision = run_initial_collision(
            dmr.qpos,
            dmr.seconds,
            robot_id=robot.robot_id,
            fps=dmr.fps,
            progress=collision_progress,
            backend=backend_selection,
        )
        stage_completed(
            PipelineStage.INITIAL_COLLISION,
            "Initial arm self-collision refinement completed.",
            metrics={"output_violations": float(initial_collision.diagnostics.output_violations)},
        )

        stage_started(
            PipelineStage.EXTRACTING_TRAJECTORIES,
            "Extracting robot root, ankle, and toe trajectories.",
        )
        targets = run_target_trajectories(
            dmr.qpos,
            initial_collision.qpos,
            dmr.seconds,
            robot_id=robot.robot_id,
            fps=dmr.fps,
        )
        stage_completed(
            PipelineStage.EXTRACTING_TRAJECTORIES,
            "Robot target trajectories extracted.",
            metrics={"frame_count": float(len(targets.seconds))},
        )

        stage_started(PipelineStage.ARA, "Running affine root adjustment.")
        ara = run_ara(targets, contacts, robot_id=robot.robot_id)
        stage_completed(
            PipelineStage.ARA,
            "Affine root adjustment completed.",
            metrics={"frame_count": float(len(ara.seconds))},
        )

        stage_started(
            PipelineStage.FPA_TARGETS,
            "Building contact-aware foot-placement targets.",
        )
        fpa_targets = build_fpa_targets(
            initial_collision.qpos,
            targets,
            cast(AraResultLike, ara),
            contacts,
            robot_id=robot.robot_id,
            fps=dmr.fps,
            source_provider=source_provider,
        )
        stage_completed(
            PipelineStage.FPA_TARGETS,
            "Contact-aware foot-placement targets completed.",
            metrics={"frame_count": float(len(fpa_targets.seconds))},
        )

        stage_started(
            PipelineStage.FPA_IK,
            "Solving foot-placement IK, recovery, and ground correction.",
        )
        fpa_ik = solve_fpa(
            fpa_targets,
            targets,
            contacts,
            robot_id=robot.robot_id,
            fps=dmr.fps,
            backend=backend_selection,
            source_provider=source_provider,
            left_ankle_target_rotations=getattr(dmr, "left_ankle_target_rotations", None),
            right_ankle_target_rotations=getattr(dmr, "right_ankle_target_rotations", None),
        )
        stage_completed(
            PipelineStage.FPA_IK,
            "Foot-placement IK and ground correction completed.",
            metrics={"frame_count": float(len(fpa_ik.seconds))},
        )

        preview_contacts = build_preview_contact_state(
            motion,
            qpos=fpa_ik.qpos,
            robot_id=robot.robot_id,
            contact_schedule=contacts,
        )

        def final_collision_progress(
            outer_pass: int,
            current: int,
            total: int,
            margin: float,
        ) -> None:
            _emit(
                sink,
                event_type=PipelineEventType.PROGRESS,
                stage=PipelineStage.FINAL_COLLISION,
                job_id=job_id,
                robot_id=robot.robot_id,
                message=f"Final collision pass {outer_pass}.",
                current=current,
                total=total,
                metrics={"margin_m": float(margin)},
            )

        stage_started(
            PipelineStage.FINAL_COLLISION,
            "Running final arm self-collision refinement.",
        )
        final_collision = run_final_collision(
            fpa_ik.qpos,
            fpa_ik.seconds,
            robot_id=robot.robot_id,
            fps=fpa_ik.fps,
            contact_labels=preview_contacts.labels,
            contact_confidence=preview_contacts.confidence,
            flight_labels=preview_contacts.flight,
            progress=final_collision_progress,
            backend=backend_selection,
        )
        stage_completed(
            PipelineStage.FINAL_COLLISION,
            "Final arm self-collision refinement completed.",
            metrics={"output_violations": float(final_collision.diagnostics.output_violations)},
        )

        stage_started(
            PipelineStage.VALIDATING_OUTPUT,
            "Computing final trajectories and ground diagnostics.",
        )
        diagnostics = run_diagnostic_trajectories(
            fpa_ik.qpos,
            final_collision.qpos,
            final_collision.seconds,
            robot_id=robot.robot_id,
            fps=final_collision.fps,
        )

        ground_diagnostics = {
            "right_support": _ground_summary(
                fpa_ik.right_ground_distance_post,
                fpa_ik.right_contact_weight,
            ),
            "left_support": _ground_summary(
                fpa_ik.left_ground_distance_post,
                fpa_ik.left_contact_weight,
            ),
        }
        support_maximum = max(
            float(ground_diagnostics["right_support"]["max_m"]),
            float(ground_diagnostics["left_support"]["max_m"]),
        )
        stage_completed(
            PipelineStage.VALIDATING_OUTPUT,
            "Final trajectory and ground diagnostics completed.",
            metrics={"support_max_clearance_m": support_maximum},
        )

        video_path = temporary / "preview" / "final.mp4" if render_video else None
        thumbnail_path = temporary / "preview" / "final.png" if render_thumbnail else None
        preview = None
        if video_path is not None or thumbnail_path is not None:
            stage_started(PipelineStage.RENDERING, "Rendering the final motion preview.")
            preview = render_motion_preview(
                robot_id=robot.robot_id,
                qpos=final_collision.qpos,
                fps=final_collision.fps,
                motion_name=source.stem,
                contacts=preview_contacts,
                video_path=video_path,
                thumbnail_path=thumbnail_path,
                width=width,
                height=height,
                source_provider=source_provider,
            )
            stage_completed(
                PipelineStage.RENDERING,
                "Final motion preview rendered.",
                metrics={"frame_count": float(preview.frame_count)},
            )

        stage_started(
            PipelineStage.EXPORTING,
            "Writing and atomically publishing the result bundle.",
        )

        stage_results: tuple[tuple[str, Any], ...] = (
            ("1_contacts", contacts),
            ("2_dmr", dmr),
            ("3_initial_collision", initial_collision),
            ("4_target_trajectories", targets),
            ("5_ara", ara),
            ("6_fpa_targets", fpa_targets),
            ("7_fpa_ik", fpa_ik),
            ("8_final", final_collision),
            ("9_diagnostics", diagnostics),
        )
        if save_stages:
            for stage_name, result in stage_results:
                path = temporary / "stages" / f"{stage_name}.npz"
                _write_npz(path, result.reference_arrays())
                stage_paths[stage_name] = path

        final_motion_path = temporary / "final" / "robot_motion.npz"
        write_robot_motion_npz(
            final_motion_path,
            robot_id=robot.robot_id,
            qpos=final_collision.qpos,
            seconds=final_collision.seconds,
            fps=final_collision.fps,
            contact_labels=preview_contacts.labels,
            contact_confidence=preview_contacts.confidence,
            contact_availability=preview_contacts.availability,
            flight_labels=preview_contacts.flight,
            source_motion_sha256=motion.sha256,
            contact_source=preview_contacts.contact_source,
            hand_contact_source=preview_contacts.hand_contact_source,
        )

        stage_artifacts = {
            name: _artifact_record(path, temporary) for name, path in stage_paths.items()
        }
        artifacts: dict[str, Any] = {
            "final_motion": _artifact_record(final_motion_path, temporary),
            "stages": stage_artifacts,
        }
        if preview is not None:
            preview_record: dict[str, Any] = {
                "width": preview.width,
                "height": preview.height,
                "fps": preview.fps,
                "frame_count": preview.frame_count,
                "camera": asdict(preview.camera),
                "style": preview.visualization_style,
                "contact_overlay": preview.contact_overlay,
                "trajectory_stage": "final",
            }
            if preview.video_path is not None:
                preview_record["video"] = _artifact_record(preview.video_path, temporary)
            if preview.thumbnail_path is not None:
                preview_record["thumbnail"] = _artifact_record(preview.thumbnail_path, temporary)
            artifacts["preview"] = preview_record

        scene_path = root_path() / robot.scene_relpath
        fpa_solver_diagnostics = _fpa_solver_diagnostics(
            fpa_targets.solve_records,
            fpa_ik.solve_records,
            frame_count=motion.frame_count,
        )
        warnings = [
            "This is a complete pipeline result but remains unreviewed motion.",
            "Generated motions require visual and physical review before hardware use.",
        ]
        if backend_selection.reason == "native_extension_unavailable":
            warnings.append(
                "The native accelerator was unavailable, so this run used the portable "
                "Python reference backend."
            )
        if not bool(fpa_solver_diagnostics["records_complete"]):
            warnings.append(
                "FPA trajectory solver provenance is incomplete or out of order; "
                "inspect diagnostics.fpa before use."
            )
        if not bool(fpa_solver_diagnostics["all_reference_solver"]):
            warnings.append(
                "FPA trajectory optimization used a solver other than CLARABEL; "
                "inspect diagnostics.fpa before use."
            )
        if not bool(fpa_solver_diagnostics["all_optimal"]):
            warnings.append(
                "FPA trajectory optimization returned a status other than optimal; "
                "inspect diagnostics.fpa before use."
            )
        if support_maximum > 0.01:
            warnings.append(
                "Support-foot clearance exceeds 10 mm "
                "in at least one frame; inspect the final video before use."
            )

        manifest: dict[str, Any] = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "classification": _CLASSIFICATION,
            "review_status": "unreviewed",
            "pipeline_complete": True,
            "last_completed_stage": "final_collision_and_diagnostics",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "core_version": __version__,
            "compute_backend": backend_selection.manifest_record(),
            "source_motion": {
                "path": source.name,
                "sha256": motion.sha256,
                "frame_count": motion.frame_count,
                "fps": motion.fps,
                "duration_seconds": motion.duration_seconds,
                "container_format": source_container,
                "provider": source_provider,
            },
            "robot": {
                "id": robot.robot_id,
                "manufacturer": robot.manufacturer,
                "model": robot.display_name,
                "model_sha256": robot.model_sha256,
                "scene_sha256": _sha256(scene_path),
            },
            "environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": _dependency_version("scipy"),
                "torch": _dependency_version("torch"),
                "mujoco": _dependency_version("mujoco"),
                "cvxpy": _dependency_version("cvxpy"),
                "clarabel": _dependency_version("clarabel"),
                "osqp": _dependency_version("osqp"),
                "ecos": _dependency_version("ecos"),
                "scs": _dependency_version("scs"),
            },
            "diagnostics": {
                "ara": asdict(ara.diagnostics),
                "fpa": fpa_solver_diagnostics,
                "initial_collision": asdict(initial_collision.diagnostics),
                "final_collision": asdict(final_collision.diagnostics),
                "ground": ground_diagnostics,
            },
            "artifacts": artifacts,
            "warnings": warnings,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_strict_json_bytes(manifest))

        if destination.exists():
            raise ArtifactError(f"Retarget output appeared during the run: {destination}")
        temporary.chmod(0o755)
        temporary.replace(destination)
        published = True
        _emit_safely(
            sink,
            event_type=PipelineEventType.STAGE_COMPLETED,
            stage=PipelineStage.EXPORTING,
            job_id=job_id,
            robot_id=robot.robot_id,
            message="Result bundle was atomically published.",
            metrics={"artifact_count": float(len(stage_paths) + 1)},
        )
    except Exception as error:
        _emit_failure_safely(
            sink,
            stage=active_stage,
            job_id=job_id,
            robot_id=robot.robot_id,
            error=error,
        )
        raise
    finally:
        if temporary is not None and not published:
            shutil.rmtree(temporary, ignore_errors=True)

    assert temporary is not None
    assert final_motion_path is not None
    published_stage_paths = {
        name: destination / path.relative_to(temporary) for name, path in stage_paths.items()
    }
    _emit_safely(
        sink,
        event_type=PipelineEventType.COMPLETED,
        stage=PipelineStage.SUCCEEDED_WITH_WARNINGS,
        job_id=job_id,
        robot_id=robot.robot_id,
        message="Complete unreviewed candidate motion was published.",
    )
    return RetargetRunResult(
        robot_id=robot.robot_id,
        output_dir=destination,
        manifest_path=destination / "manifest.json",
        final_motion_path=destination / final_motion_path.relative_to(temporary),
        stage_paths=published_stage_paths,
        video_path=(
            destination / video_path.relative_to(temporary) if video_path is not None else None
        ),
        thumbnail_path=(
            destination / thumbnail_path.relative_to(temporary)
            if thumbnail_path is not None
            else None
        ),
    )


__all__ = ["RetargetRunResult", "run_retarget_pipeline"]
