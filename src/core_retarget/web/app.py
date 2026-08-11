"""FastAPI application for the CoRe browser demo."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core_retarget._version import __version__
from core_retarget.exceptions import CoReError
from core_retarget.motion.source import SourceMotionSummary, validate_source_motion
from core_retarget.native import resolve_backend
from core_retarget.robots.registry import get_robot, list_robots
from core_retarget.web.jobs import (
    SOURCE_MOTION_SUFFIXES,
    ArtifactNotFoundError,
    JobCapacityError,
    JobManager,
    JobNotFoundError,
)


@dataclass(frozen=True, slots=True)
class WebConfig:
    runs_dir: Path = Path("runs/web")
    max_upload_bytes: int = 256 * 1024 * 1024
    max_frames: int = 1_000_000
    max_active_jobs: int | None = None
    result_ttl_seconds: float | None = None
    max_video_width: int = 3840
    max_video_height: int = 2160
    allow_stage_archives: bool = True

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive.")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        if self.max_active_jobs is not None and self.max_active_jobs <= 0:
            raise ValueError("max_active_jobs must be positive when configured.")
        if self.result_ttl_seconds is not None and self.result_ttl_seconds <= 0.0:
            raise ValueError("result_ttl_seconds must be positive when configured.")
        if self.max_video_width < 320 or self.max_video_height < 240:
            raise ValueError("Video limits must be at least 320×240.")


def _summary_payload(summary: SourceMotionSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["path"] = summary.path.name
    payload["keys"] = list(summary.keys)
    payload["warnings"] = list(summary.warnings)
    return payload


def _job_payload(snapshot: dict[str, Any] | Any) -> dict[str, Any]:
    payload = dict(snapshot)
    job_id = str(payload["job_id"])
    payload["artifacts"] = {
        name: {
            "url": f"/api/jobs/{job_id}/artifacts/{name}",
            "label": {
                "motion": "Robot motion NPZ",
                "manifest": "Run manifest",
                "video": "Result MP4",
                "thumbnail": "Preview image",
            }.get(name, name),
        }
        for name in payload.get("artifacts", [])
    }
    return payload


def _safe_original_filename(upload: UploadFile) -> str:
    filename = Path(upload.filename or "source_motion.npz").name
    if Path(filename).suffix.lower() not in SOURCE_MOTION_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="CoRe accepts Kimodo NPZ or GEM-X PT source motion.",
        )
    return filename


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("xb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    )
                stream.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded motion is empty.")
    return total


def create_app(
    config: WebConfig | None = None,
    *,
    manager: JobManager | None = None,
) -> FastAPI:
    """Create the same-origin CoRe web application."""

    settings = config or WebConfig()
    backend = resolve_backend("auto")
    owns_manager = manager is None
    jobs = manager or JobManager(
        settings.runs_dir,
        max_active_jobs=settings.max_active_jobs,
        result_ttl_seconds=settings.result_ttl_seconds,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        cleanup_task: asyncio.Task[None] | None = None
        if jobs.result_ttl_seconds is not None:
            interval = min(60.0, jobs.result_ttl_seconds)

            async def cleanup_expired_jobs() -> None:
                while True:
                    await asyncio.sleep(interval)
                    jobs.purge_expired()

            cleanup_task = asyncio.create_task(cleanup_expired_jobs())
        try:
            yield
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup_task
            if owns_manager:
                jobs.shutdown(wait=True)

    app = FastAPI(
        title="CoRe Web Demo",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.jobs = jobs
    app.state.web_config = settings
    app.state.compute_backend = backend

    static_dir = Path(str(files("core_retarget.web").joinpath("static")))
    index_path = Path(
        str(files("core_retarget.web").joinpath("templates").joinpath("index.html"))
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": backend.selected,
            "backend_reason": backend.reason,
            "source_formats": ["npz", "pt"],
            "limits": {
                "max_upload_bytes": settings.max_upload_bytes,
                "max_frames": settings.max_frames,
                "max_active_jobs": settings.max_active_jobs,
                "result_ttl_seconds": settings.result_ttl_seconds,
                "max_video_width": settings.max_video_width,
                "max_video_height": settings.max_video_height,
                "allow_stage_archives": settings.allow_stage_archives,
            },
        }

    @app.get("/api/robots")
    async def robots() -> dict[str, Any]:
        return {
            "robots": [
                {
                    "id": robot.robot_id,
                    "name": robot.display_name,
                    "manufacturer": robot.manufacturer,
                    "dof": robot.actuated_dof,
                }
                for robot in list_robots()
            ]
        }

    @app.post("/api/motions/validate")
    async def validate_motion(
        motion: Annotated[UploadFile, File(...)],
        fps: Annotated[float | None, Form()] = None,
    ) -> dict[str, Any]:
        original_filename = _safe_original_filename(motion)
        source_suffix = Path(original_filename).suffix.lower()
        job_id, input_path, _output_dir = jobs.allocate(source_suffix)
        try:
            size_bytes = await _save_upload(motion, input_path, settings.max_upload_bytes)
            summary = validate_source_motion(
                input_path,
                fps_override=fps,
                max_file_bytes=settings.max_upload_bytes,
                max_frames=settings.max_frames,
            )
            payload = _summary_payload(summary)
            payload.update(
                {
                    "valid": True,
                    "filename": original_filename,
                    "size_bytes": size_bytes,
                }
            )
            return payload
        except HTTPException:
            raise
        except CoReError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            jobs.discard_allocation(job_id)

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        motion: Annotated[UploadFile, File(...)],
        robot: Annotated[str, Form()],
        render_video: Annotated[bool, Form()] = True,
        save_stages: Annotated[bool, Form()] = False,
        fps: Annotated[float | None, Form()] = None,
        width: Annotated[int, Form()] = 1280,
        height: Annotated[int, Form()] = 720,
    ) -> dict[str, Any]:
        original_filename = _safe_original_filename(motion)
        try:
            robot_id = get_robot(robot).robot_id
        except CoReError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if (
            width < 320
            or width > settings.max_video_width
            or height < 240
            or height > settings.max_video_height
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Video dimensions must be between 320×240 and "
                    f"{settings.max_video_width}×{settings.max_video_height}."
                ),
            )
        if save_stages and not settings.allow_stage_archives:
            raise HTTPException(
                status_code=422,
                detail="Intermediate stage archives are disabled on this deployment.",
            )
        try:
            jobs.ensure_capacity()
        except JobCapacityError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "30"},
            ) from exc

        source_suffix = Path(original_filename).suffix.lower()
        job_id, input_path, output_dir = jobs.allocate(source_suffix)
        submitted = False
        try:
            await _save_upload(motion, input_path, settings.max_upload_bytes)
            validate_source_motion(
                input_path,
                fps_override=fps,
                max_file_bytes=settings.max_upload_bytes,
                max_frames=settings.max_frames,
            )
            snapshot = jobs.submit(
                job_id=job_id,
                original_filename=original_filename,
                input_path=input_path,
                output_dir=output_dir,
                robot_id=robot_id,
                render_video=render_video,
                save_stages=save_stages,
                fps_override=fps,
                width=width,
                height=height,
            )
            submitted = True
            return _job_payload(snapshot)
        except HTTPException:
            raise
        except JobCapacityError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "30"},
            ) from exc
        except CoReError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if not submitted:
                jobs.discard_allocation(job_id)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _job_payload(jobs.get(job_id))
        except (JobNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Unknown CoRe job.") from exc

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        request: Request,
        job_id: str,
        after: int = 0,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        try:
            if last_event_id is not None:
                after = max(after, int(last_event_id))
            jobs.get(job_id)
        except (JobNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Unknown CoRe job.") from exc

        async def stream() -> AsyncIterator[str]:
            sequence = after
            while True:
                if await request.is_disconnected():
                    break
                events = jobs.events_after(job_id, sequence)
                for event in events:
                    sequence = int(event["sequence"])
                    yield (
                        f"id: {sequence}\n"
                        "event: message\n"
                        f"data: {json.dumps(event, ensure_ascii=False, allow_nan=False)}\n\n"
                    )
                snapshot = jobs.get(job_id)
                if snapshot["status"] in {"succeeded", "failed"} and not events:
                    break
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
    async def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        try:
            path = jobs.artifact(job_id, artifact_name)
        except (JobNotFoundError, ArtifactNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Unknown CoRe artifact.") from exc
        media_types = {
            "motion": "application/octet-stream",
            "manifest": "application/json",
            "video": "video/mp4",
            "thumbnail": "image/png",
        }
        filenames = {
            "motion": f"core-{job_id[:8]}-robot-motion.npz",
            "manifest": f"core-{job_id[:8]}-manifest.json",
            "video": f"core-{job_id[:8]}-preview.mp4",
            "thumbnail": f"core-{job_id[:8]}-preview.png",
        }
        return FileResponse(
            path,
            media_type=media_types.get(artifact_name, "application/octet-stream"),
            filename=filenames.get(artifact_name, path.name),
            content_disposition_type=(
                "inline" if artifact_name in {"video", "thumbnail"} else "attachment"
            ),
        )

    return app


__all__ = ["WebConfig", "create_app"]
