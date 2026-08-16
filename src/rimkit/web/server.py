"""Uvicorn entry point used by ``rimkit serve``."""

from __future__ import annotations

from pathlib import Path


def serve(
    *,
    host: str,
    port: int,
    runs_dir: Path,
    max_upload_mb: int,
    max_frames: int,
    max_active_jobs: int,
    result_ttl_minutes: int,
    max_video_width: int,
    max_video_height: int,
    allow_stage_archives: bool,
) -> None:
    import uvicorn

    from rimkit.web.app import WebConfig, create_app

    app = create_app(
        WebConfig(
            runs_dir=runs_dir,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_frames=max_frames,
            max_active_jobs=max_active_jobs or None,
            result_ttl_seconds=(result_ttl_minutes * 60.0 or None),
            max_video_width=max_video_width,
            max_video_height=max_video_height,
            allow_stage_archives=allow_stage_archives,
        )
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["serve"]
