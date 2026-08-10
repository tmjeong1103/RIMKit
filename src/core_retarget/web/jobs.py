"""Thread-safe job queue for the browser adapter."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from core_retarget.pipeline.events import CallbackEventSink, PipelineEvent
from core_retarget.pipeline.runner import RetargetRunResult, run_retarget_pipeline

LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED})


class PipelineRunner(Protocol):
    def __call__(
        self,
        motion_path: str | Path,
        robot_id: str,
        output_dir: str | Path,
        *,
        fps_override: float | None = None,
        save_stages: bool = True,
        render_video: bool = False,
        render_thumbnail: bool = False,
        width: int = 1280,
        height: int = 720,
        event_sink: CallbackEventSink | None = None,
    ) -> RetargetRunResult: ...


@dataclass
class _Job:
    job_id: str
    original_filename: str
    input_path: Path
    output_dir: Path
    robot_id: str
    render_video: bool
    save_stages: bool
    fps_override: float | None
    width: int
    height: int
    status: JobStatus = JobStatus.QUEUED
    stage: str = "CREATED"
    message: str = "Waiting for the CoRe worker."
    current: int | None = None
    total: int | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    finished_monotonic: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


class JobNotFoundError(KeyError):
    """Raised when a web job ID is unknown."""


class ArtifactNotFoundError(KeyError):
    """Raised when a job does not expose the requested artifact."""


class JobCapacityError(RuntimeError):
    """Raised when the configured web queue has no free capacity."""


class JobManager:
    """Run CoRe jobs serially while exposing immutable JSON snapshots."""

    def __init__(
        self,
        runs_dir: str | Path,
        *,
        max_workers: int = 1,
        max_active_jobs: int | None = None,
        result_ttl_seconds: float | None = None,
        runner: PipelineRunner = run_retarget_pipeline,
    ) -> None:
        if max_workers != 1:
            raise ValueError("The CoRe web adapter currently supports one worker.")
        if max_active_jobs is not None and max_active_jobs <= 0:
            raise ValueError("max_active_jobs must be positive when configured.")
        if result_ttl_seconds is not None and result_ttl_seconds <= 0.0:
            raise ValueError("result_ttl_seconds must be positive when configured.")
        self.runs_dir = Path(runs_dir).expanduser().resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_active_jobs = max_active_jobs
        self.result_ttl_seconds = result_ttl_seconds
        self._runner = runner
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="core-web",
        )
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._closed = False

    def allocate(self) -> tuple[str, Path, Path]:
        """Reserve a server-owned job directory for one uploaded motion."""

        with self._lock:
            if self._closed:
                raise RuntimeError("The web job manager is closed.")
            self._purge_expired_locked()
            while True:
                job_id = uuid.uuid4().hex
                job_root = self.runs_dir / job_id
                try:
                    (job_root / "input").mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                return job_id, job_root / "input" / "source_motion.npz", job_root / "result"

    def discard_allocation(self, job_id: str) -> None:
        """Remove an unsubmitted, empty-or-upload-only allocation."""

        with self._lock:
            if job_id in self._jobs:
                raise RuntimeError("Cannot discard a submitted job.")
        target = self._job_root(job_id)
        if target.is_dir():
            shutil.rmtree(target)

    def submit(
        self,
        *,
        job_id: str,
        original_filename: str,
        input_path: Path,
        output_dir: Path,
        robot_id: str,
        render_video: bool,
        save_stages: bool,
        fps_override: float | None,
        width: int,
        height: int,
    ) -> Mapping[str, Any]:
        input_path = input_path.resolve()
        output_dir = output_dir.resolve()
        expected_root = self._job_root(job_id)
        if input_path.parent.parent != expected_root or output_dir.parent != expected_root:
            raise ValueError("Web jobs must stay inside their allocated directory.")
        if not input_path.is_file():
            raise ValueError("The uploaded source motion is missing.")

        job = _Job(
            job_id=job_id,
            original_filename=original_filename,
            input_path=input_path,
            output_dir=output_dir,
            robot_id=robot_id,
            render_video=render_video,
            save_stages=save_stages,
            fps_override=fps_override,
            width=width,
            height=height,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("The web job manager is closed.")
            self._purge_expired_locked()
            self._ensure_capacity_locked()
            if job_id in self._jobs:
                raise ValueError(f"Duplicate web job ID: {job_id}")
            self._jobs[job_id] = job
            self._append_event(
                job,
                {
                    "event_type": "job_queued",
                    "stage": job.stage,
                    "message": job.message,
                    "timestamp": job.created_at,
                },
            )
            snapshot = self._snapshot(job)
            self._executor.submit(self._execute, job_id)
            return snapshot

    def get(self, job_id: str) -> Mapping[str, Any]:
        with self._lock:
            self._purge_expired_locked()
            return self._snapshot(self._get_job(job_id))

    def ensure_capacity(self) -> None:
        """Reject a new submission before its upload when the queue is full."""

        with self._lock:
            self._purge_expired_locked()
            self._ensure_capacity_locked()

    def purge_expired(self) -> None:
        """Remove terminal jobs whose configured retention period has elapsed."""

        with self._lock:
            self._purge_expired_locked()

    def events_after(self, job_id: str, sequence: int) -> list[dict[str, Any]]:
        if sequence < 0:
            raise ValueError("Event sequence cannot be negative.")
        with self._lock:
            self._purge_expired_locked()
            job = self._get_job(job_id)
            return [dict(event) for event in job.events if int(event["sequence"]) > sequence]

    def wait_for_terminal(self, job_id: str, *, timeout: float) -> Mapping[str, Any]:
        """Wait for a job in tests and non-async adapters."""

        with self._condition:
            finished = self._condition.wait_for(
                lambda: self._get_job(job_id).status in TERMINAL_STATUSES,
                timeout=timeout,
            )
            if not finished:
                raise TimeoutError(f"Web job did not finish within {timeout:g} seconds.")
            return self._snapshot(self._get_job(job_id))

    def artifact(self, job_id: str, name: str) -> Path:
        with self._lock:
            self._purge_expired_locked()
            job = self._get_job(job_id)
            try:
                path = job.artifacts[name]
            except KeyError as exc:
                raise ArtifactNotFoundError(name) from exc
            path = path.resolve()
            if not path.is_relative_to(self._job_root(job_id)) or not path.is_file():
                raise ArtifactNotFoundError(name)
            return path

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _get_job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _ensure_capacity_locked(self) -> None:
        if self.max_active_jobs is None:
            return
        active = sum(job.status not in TERMINAL_STATUSES for job in self._jobs.values())
        if active >= self.max_active_jobs:
            raise JobCapacityError(
                f"CoRe is currently processing or queueing {active} job(s); try again later."
            )

    def _purge_expired_locked(self) -> None:
        if self.result_ttl_seconds is None:
            return
        cutoff = time.monotonic() - self.result_ttl_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_monotonic is not None and job.finished_monotonic <= cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]
            shutil.rmtree(self._job_root(job_id), ignore_errors=True)

    def _job_root(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise ValueError("Invalid web job ID.")
        return (self.runs_dir / job_id).resolve()

    def _append_event(self, job: _Job, payload: Mapping[str, Any]) -> None:
        event = dict(payload)
        event["job_id"] = job.job_id
        event["robot_id"] = job.robot_id
        event["sequence"] = len(job.events) + 1
        job.events.append(event)
        self._condition.notify_all()

    def _record_pipeline_event(self, job_id: str, event: PipelineEvent) -> None:
        with self._condition:
            job = self._get_job(job_id)
            payload = event.to_dict()
            payload["pipeline_job_id"] = payload.pop("job_id")
            job.stage = event.stage.value
            job.message = event.message
            job.current = event.current
            job.total = event.total
            if event.event_type.value == "warning":
                job.warnings.append(event.message)
            self._append_event(job, payload)

    def _execute(self, job_id: str) -> None:
        with self._condition:
            job = self._get_job(job_id)
            job.status = JobStatus.RUNNING
            job.started_at = _now()
            job.message = "CoRe retargeting started."
            self._append_event(
                job,
                {
                    "event_type": "job_started",
                    "stage": job.stage,
                    "message": job.message,
                    "timestamp": job.started_at,
                },
            )

        try:
            result = self._runner(
                job.input_path,
                job.robot_id,
                job.output_dir,
                fps_override=job.fps_override,
                save_stages=job.save_stages,
                render_video=job.render_video,
                render_thumbnail=job.render_video,
                width=job.width,
                height=job.height,
                event_sink=CallbackEventSink(
                    lambda event: self._record_pipeline_event(job_id, event)
                ),
            )
        except Exception as exc:
            LOGGER.exception("CoRe web job %s failed", job_id)
            with self._condition:
                job = self._get_job(job_id)
                job.status = JobStatus.FAILED
                job.stage = "FAILED"
                job.error = str(exc).strip() or type(exc).__name__
                job.message = "Retargeting failed."
                job.finished_at = _now()
                job.finished_monotonic = time.monotonic()
                self._append_event(
                    job,
                    {
                        "event_type": "job_failed",
                        "stage": job.stage,
                        "message": job.message,
                        "error": job.error,
                        "timestamp": job.finished_at,
                    },
                )
            return

        with self._condition:
            job = self._get_job(job_id)
            job.status = JobStatus.SUCCEEDED
            job.stage = "SUCCEEDED_WITH_WARNINGS"
            job.message = "Robot motion and preview are ready."
            job.finished_at = _now()
            job.finished_monotonic = time.monotonic()
            job.current = None
            job.total = None
            job.artifacts = self._result_artifacts(result)
            self._append_event(
                job,
                {
                    "event_type": "job_succeeded",
                    "stage": job.stage,
                    "message": job.message,
                    "timestamp": job.finished_at,
                    "artifacts": sorted(job.artifacts),
                },
            )

    @staticmethod
    def _result_artifacts(result: RetargetRunResult) -> dict[str, Path]:
        artifacts = {
            "motion": result.final_motion_path,
            "manifest": result.manifest_path,
        }
        if result.video_path is not None:
            artifacts["video"] = result.video_path
        if result.thumbnail_path is not None:
            artifacts["thumbnail"] = result.thumbnail_path
        return artifacts

    @staticmethod
    def _snapshot(job: _Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "stage": job.stage,
            "message": job.message,
            "current": job.current,
            "total": job.total,
            "robot_id": job.robot_id,
            "source_filename": job.original_filename,
            "render_video": job.render_video,
            "save_stages": job.save_stages,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "warnings": list(job.warnings),
            "artifacts": sorted(job.artifacts),
            "last_event_sequence": len(job.events),
        }


__all__ = [
    "ArtifactNotFoundError",
    "JobCapacityError",
    "JobManager",
    "JobNotFoundError",
    "JobStatus",
    "TERMINAL_STATUSES",
]
