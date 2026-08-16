"""RIMKit command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rimkit import __version__
from rimkit.api import Retargeter
from rimkit.config.schema import RunConfig
from rimkit.exceptions import PipelineNotAvailableError, RIMKitError
from rimkit.methods import get_method, list_methods
from rimkit.motion.source import validate_source_motion
from rimkit.native import resolve_backend
from rimkit.pipeline.events import (
    CallbackEventSink,
    PipelineEvent,
    PipelineEventType,
)
from rimkit.review import run_review
from rimkit.robots.profiles import get_initial_collision_profile
from rimkit.robots.registry import get_robot, list_robots
from rimkit.robots.validation import verify_robot


def _robot_rows() -> list[dict[str, object]]:
    return [
        {
            "id": robot.robot_id,
            "manufacturer": robot.manufacturer,
            "model": robot.display_name,
            "dof": robot.actuated_dof,
            "license": robot.license_spdx,
        }
        for robot in list_robots()
    ]


def _robot_choices() -> tuple[str, ...]:
    return tuple(robot.robot_id for robot in list_robots())


def _method_rows() -> list[dict[str, str]]:
    return [
        {
            "id": method.method_id,
            "name": method.display_name,
            "description": method.description,
        }
        for method in list_methods()
    ]


def _run_methods(args: argparse.Namespace) -> int:
    rows = _method_rows()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['id']:<8} {row['name']:<8} {row['description']}")
    return 0


def _print_robot_table() -> None:
    print(f"{'ID':<4} {'Manufacturer':<18} {'Model':<12} {'DOF':>4}  License")
    for row in _robot_rows():
        print(
            f"{row['id']:<4} {row['manufacturer']:<18} {row['model']:<12} "
            f"{row['dof']:>4}  {row['license']}"
        )


def _run_robots(args: argparse.Namespace) -> int:
    if args.robots_command == "list":
        if args.json:
            print(json.dumps(_robot_rows(), indent=2))
        else:
            _print_robot_table()
        return 0

    selected = [get_robot(args.robot)] if args.robot else list(list_robots())
    results = [verify_robot(robot, load_mujoco=not args.static_only) for robot in selected]
    if args.json:
        payload = [
            {
                "robot_id": result.robot_id,
                "ok": result.ok,
                "model_info": result.model_info,
                "issues": [asdict(issue) for issue in result.issues],
            }
            for result in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            state = "OK" if result.ok else "FAILED"
            info = " ".join(f"{key}={value}" for key, value in result.model_info.items())
            print(f"[{state}] {result.robot_id}" + (f"  {info}" if info else ""))
            for issue in result.issues:
                print(f"  {issue.severity.upper()} {issue.code}: {issue.message}")
    return 0 if all(result.ok for result in results) else 1


def _run_validate(args: argparse.Namespace) -> int:
    summary = validate_source_motion(args.motion, fps_override=args.fps)
    if args.json:
        payload = asdict(summary)
        payload["path"] = str(payload["path"])
        print(json.dumps(payload, indent=2))
        return 0

    contacts = (
        f"{summary.contact_channels} channels"
        if summary.contact_channels is not None
        else "not provided"
    )
    print(f"Input: {summary.path}")
    print(f"Source: {summary.provider} ({summary.container_format.upper()})")
    print(f"SHA-256: {summary.sha256}")
    print(f"Frames: {summary.frame_count}")
    print(f"FPS: {summary.fps:g}")
    print(f"Duration: {summary.duration_seconds:.3f} s")
    print(f"Foot contacts: {contacts}")
    print(f"Keys: {', '.join(summary.keys)}")
    for warning in summary.warnings:
        print(f"WARNING: {warning}")
    print("Status: valid")
    return 0


def _run_backend(args: argparse.Namespace) -> int:
    selection = resolve_backend("native" if args.require_native else "auto")
    payload = selection.manifest_record(include_detail=True)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Requested: {selection.requested}")
        print(f"Selected: {selection.selected}")
        print(f"Reason: {selection.reason}")
        if selection.module_name is not None:
            print(f"Module: {selection.module_name}")
        for name, value in selection.native_info.items():
            print(f"{name}: {value}")
        if selection.detail is not None:
            print(f"Loader detail: {selection.detail}")
    return 0


def _run_review(args: argparse.Namespace) -> int:
    robot_id = get_robot(args.robot).robot_id
    collision_passes = get_initial_collision_profile(robot_id).outer_passes
    output_dir = args.output / args.motion.stem / robot_id

    def dmr_progress(current: int, total: int, error: float) -> None:
        if current == 1 or current == total or current % 30 == 0:
            print(
                f"DMR {current}/{total} error={error:.6g}",
                file=sys.stderr,
                flush=True,
            )

    def collision_progress(
        outer_pass: int,
        current: int,
        total: int,
        margin: float,
    ) -> None:
        if current == 1 or current == total or current % 30 == 0:
            print(
                f"INITIAL_COLLISION pass={outer_pass}/{collision_passes} "
                f"frame={current}/{total} "
                f"margin={margin:.3f}",
                file=sys.stderr,
                flush=True,
            )

    result = run_review(
        args.motion,
        robot_id,
        output_dir,
        render_video=args.video,
        render_thumbnail=args.thumbnail,
        width=args.width,
        height=args.height,
        fps_override=args.fps,
        dmr_progress=dmr_progress,
        collision_progress=collision_progress,
        backend=args.backend,
    )
    payload = {
        "classification": "stage3-review",
        "review_status": "unreviewed",
        "pipeline_complete": False,
        "output_dir": str(result.output_dir),
        "manifest": str(result.manifest_path),
        "contacts": str(result.contacts_path),
        "dmr": str(result.dmr_path),
        "initial_collision": str(result.initial_collision_path),
        "video": str(result.video_path) if result.video_path is not None else None,
        "thumbnail": (str(result.thumbnail_path) if result.thumbnail_path is not None else None),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Review artifacts generated (unreviewed; pipeline stops at Stage 3).")
        for name, path in payload.items():
            if name not in {"classification", "review_status", "pipeline_complete"} and path:
                print(f"{name}: {path}")
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    method = get_method(args.method)
    robot_id = get_robot(args.robot).robot_id
    output_dir = args.output / args.motion.stem / robot_id

    def print_event(event: PipelineEvent) -> None:
        if event.event_type == PipelineEventType.PROGRESS:
            current = event.current
            total = event.total
            if (
                current is not None
                and total is not None
                and current not in (1, total)
                and current % 30 != 0
            ):
                return
            progress = "" if current is None or total is None else f" {current}/{total}"
            metrics = " ".join(f"{name}={value:.6g}" for name, value in event.metrics.items())
            print(
                f"{event.stage.value}{progress} {event.message} {metrics}".rstrip(),
                file=sys.stderr,
                flush=True,
            )
        elif event.event_type in {
            PipelineEventType.WARNING,
            PipelineEventType.FAILED,
            PipelineEventType.COMPLETED,
        }:
            print(
                f"{event.stage.value}: {event.message}",
                file=sys.stderr,
                flush=True,
            )

    result = Retargeter(
        robot_id,
        RunConfig(robot=robot_id, fps=args.fps, backend=args.backend),
    ).run(
        args.motion,
        output_dir,
        save_stages=not args.no_stages,
        render_video=args.video,
        render_thumbnail=args.thumbnail,
        width=args.width,
        height=args.height,
        event_sink=CallbackEventSink(print_event),
    )
    payload = {
        "classification": "core-final-candidate",
        "method": method.method_id,
        "review_status": "unreviewed",
        "pipeline_complete": True,
        "robot_id": result.robot_id,
        "output_dir": str(result.output_dir),
        "manifest": str(result.manifest_path),
        "final_motion": str(result.final_motion_path),
        "stages": {name: str(path) for name, path in result.stage_paths.items()},
        "video": str(result.video_path) if result.video_path is not None else None,
        "thumbnail": (str(result.thumbnail_path) if result.thumbnail_path is not None else None),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Complete CoRe candidate generated (visual review still required).")
        print(f"output_dir: {result.output_dir}")
        print(f"manifest: {result.manifest_path}")
        print(f"final_motion: {result.final_motion_path}")
        if result.video_path is not None:
            print(f"video: {result.video_path}")
        if result.thumbnail_path is not None:
            print(f"thumbnail: {result.thumbnail_path}")
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    try:
        from rimkit.web.server import serve

        serve(
            host=args.host,
            port=args.port,
            runs_dir=args.runs_dir,
            max_upload_mb=args.max_upload_mb,
            max_frames=args.max_frames,
            max_active_jobs=args.max_active_jobs,
            result_ttl_minutes=args.result_ttl_minutes,
            max_video_width=args.max_video_width,
            max_video_height=args.max_video_height,
            allow_stage_archives=not args.disable_stage_archives,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "multipart", "uvicorn"}:
            raise PipelineNotAvailableError(
                'The browser interface needs optional dependencies. Install ".[web]".'
            ) from exc
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rimkit",
        description="Robot Intelligence Lab Motion Kit.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    methods_parser = commands.add_parser("methods", help="Inspect available methods.")
    method_commands = methods_parser.add_subparsers(dest="methods_command", required=True)
    methods_list = method_commands.add_parser("list", help="List available methods.")
    methods_list.add_argument("--json", action="store_true")

    robots_parser = commands.add_parser("robots", help="Inspect bundled robot models.")
    robot_commands = robots_parser.add_subparsers(dest="robots_command", required=True)
    robots_list = robot_commands.add_parser("list", help="List supported robots.")
    robots_list.add_argument("--json", action="store_true")
    robots_verify = robot_commands.add_parser(
        "verify", help="Verify model files and MuJoCo contracts."
    )
    robots_verify.add_argument("robot", nargs="?", help="Optional robot ID.")
    robots_verify.add_argument(
        "--static-only",
        action="store_true",
        help="Check files and hashes without importing MuJoCo.",
    )
    robots_verify.add_argument("--json", action="store_true")

    validate_parser = commands.add_parser(
        "validate", help="Validate a Kimodo .npz or GEM-X .pt SOMA motion."
    )
    validate_parser.add_argument("motion", type=Path)
    validate_parser.add_argument("--fps", type=float)
    validate_parser.add_argument("--json", action="store_true")

    backend_parser = commands.add_parser(
        "backend",
        help="Inspect the installed native accelerator and Python fallback.",
    )
    backend_parser.add_argument(
        "--require-native",
        action="store_true",
        help="Return an error instead of allowing the Python fallback.",
    )
    backend_parser.add_argument("--json", action="store_true")

    review_parser = commands.add_parser(
        "review",
        help="Run DMR and initial collision and save explicitly unreviewed artifacts.",
    )
    review_parser.add_argument("motion", type=Path)
    review_parser.add_argument("--robot", required=True, choices=_robot_choices())
    review_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/review"),
        help="Output root; artifacts are written below <motion>/<robot>.",
    )
    review_parser.add_argument("--fps", type=float, help="Override source FPS.")
    review_parser.add_argument(
        "--backend",
        choices=("auto", "native", "python"),
        default="auto",
        help="Compute backend (default: native when available, otherwise Python).",
    )
    review_parser.add_argument("--video", action="store_true", help="Render an MP4 preview.")
    review_parser.add_argument(
        "--thumbnail", action="store_true", help="Render a PNG preview frame."
    )
    review_parser.add_argument("--width", type=int, default=1280)
    review_parser.add_argument("--height", type=int, default=720)
    review_parser.add_argument("--json", action="store_true")

    run_parser = commands.add_parser("run", help="Run the retargeting pipeline.")
    run_parser.add_argument("motion", type=Path)
    run_parser.add_argument(
        "--method",
        choices=tuple(method.method_id for method in list_methods()),
        default="core",
        help="Retargeting method (default: core).",
    )
    run_parser.add_argument("--robot", required=True, choices=_robot_choices())
    run_parser.add_argument("--output", type=Path, default=Path("runs"))
    run_parser.add_argument("--fps", type=float, help="Override source FPS.")
    run_parser.add_argument(
        "--backend",
        choices=("auto", "native", "python"),
        default="auto",
        help="Compute backend (default: native when available, otherwise Python).",
    )
    run_parser.add_argument(
        "--no-stages",
        action="store_true",
        help="Save only the final motion, manifest, and requested media.",
    )
    run_parser.add_argument("--video", action="store_true", help="Render final MP4 output.")
    run_parser.add_argument(
        "--thumbnail", action="store_true", help="Render a final PNG thumbnail."
    )
    run_parser.add_argument("--width", type=int, default=1280)
    run_parser.add_argument("--height", type=int, default=720)
    run_parser.add_argument("--json", action="store_true")

    serve_parser = commands.add_parser("serve", help="Start the local browser interface.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--runs-dir", type=Path, default=Path("runs/web"))
    serve_parser.add_argument("--max-upload-mb", type=int, default=256)
    serve_parser.add_argument("--max-frames", type=int, default=1_000_000)
    serve_parser.add_argument(
        "--max-active-jobs",
        type=int,
        default=0,
        help="maximum running plus queued jobs; 0 keeps the local queue unlimited",
    )
    serve_parser.add_argument(
        "--result-ttl-minutes",
        type=int,
        default=0,
        help="delete completed web jobs after this many minutes; 0 retains them",
    )
    serve_parser.add_argument("--max-video-width", type=int, default=3840)
    serve_parser.add_argument("--max-video-height", type=int, default=2160)
    serve_parser.add_argument(
        "--disable-stage-archives",
        action="store_true",
        help="reject requests that ask the web worker to retain intermediate stages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            return _run_methods(args)
        if args.command == "robots":
            return _run_robots(args)
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "backend":
            return _run_backend(args)
        if args.command == "review":
            return _run_review(args)
        if args.command == "run":
            return _run_pipeline(args)
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                parser.error("--port must be between 1 and 65535")
            if args.max_upload_mb <= 0:
                parser.error("--max-upload-mb must be positive")
            if args.max_frames <= 0:
                parser.error("--max-frames must be positive")
            if args.max_active_jobs < 0:
                parser.error("--max-active-jobs must be non-negative")
            if args.result_ttl_minutes < 0:
                parser.error("--result-ttl-minutes must be non-negative")
            if args.max_video_width < 320 or args.max_video_height < 240:
                parser.error("video limits must be at least 320x240")
            return _run_serve(args)
        parser.error(f"Unknown command: {args.command}")
    except RIMKitError as exc:
        print(f"rimkit: error: {exc}", file=sys.stderr)
        return 2
    return 2
