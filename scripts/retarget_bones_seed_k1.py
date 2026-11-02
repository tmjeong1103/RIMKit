#!/usr/bin/env python3
"""Resumable K1 + video + CATO-WM export of the selected Uniform catalog."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

RIMKIT = Path(__file__).resolve().parents[1]
WORKSPACE = RIMKIT.parent
CATO = WORKSPACE / "CATO-WM"
DEFAULT_SELECTION = RIMKIT / "runs/bones_seed_selection_v3/representatives.json"
DEFAULT_OUTPUT = CATO / "resources/motions/bones-seed-k1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def complete(root, row):
    receipt = root / row["motion_group"] / "result.json"
    if not receipt.is_file():
        return False
    record = json.loads(receipt.read_text())
    if record["selection_filename"] != row["filename"]:
        raise ValueError("selection identity changed")
    return all(
        (root / p).is_file() and digest(root / p) == sha for p, sha in record["file_pins"].items()
    )


def one(row, root, backend):
    """Multiple controllers can safely consume the catalog in either order."""
    directory = root / row["motion_group"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".retarget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _one_unlocked(row, root, backend)


def _one_unlocked(row, root, backend):
    import numpy as np

    sys.path[:0] = [str(RIMKIT / "src"), str(CATO / "src")]
    from cato_updated.frozen.ai_sapiens_adapter.motion import (
        kinematic_joint_names_from_mjcf,
        load_mimickit_motion,
    )
    from cato_updated.bones_seed_export import export_optional_50hz
    from cato_updated.frozen.reference.rimkit import load_rimkit_reference

    from rimkit.motion.bvh import convert_bones_seed_bvh
    from rimkit.pipeline.events import CallbackEventSink, PipelineEventType
    from rimkit.pipeline.runner import run_retarget_pipeline

    if complete(root, row):
        return json.loads((root / row["motion_group"] / "result.json").read_text())
    started = time.monotonic()
    directory = root / row["motion_group"]
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source/soma77_30fps_contacts.npz"
    if not source.exists():
        convert_bones_seed_bvh(
            row["uniform_bvh_path"], source, target_fps=30.0, ground=True, estimate_contacts=True
        )
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["conversion_metadata"].item()))
        assert metadata["source_sha256"] == digest(row["uniform_bvh_path"])
        assert float(data["fps"]) == 30.0
        assert (
            str(data["foot_contact_source"].item()) == "bones_seed_kinematic_foot_point_contacts_v1"
        )
    run = directory / "k1"
    if not (run / "manifest.json").is_file():

        def event(e):
            if e.event_type in (
                PipelineEventType.STAGE_STARTED,
                PipelineEventType.STAGE_COMPLETED,
                PipelineEventType.FAILED,
            ):
                print(json.dumps(e.to_dict()), flush=True)

        run_retarget_pipeline(
            source,
            "k1",
            run,
            save_stages=True,
            render_video=True,
            render_thumbnail=True,
            width=854,
            height=480,
            backend=backend,
            event_sink=CallbackEventSink(event),
        )
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["source_motion"]["sha256"] == digest(source)
    assert manifest["pipeline_complete"] and manifest["robot"]["id"] == "k1"
    for item in [
        manifest["artifacts"]["final_motion"],
        *manifest["artifacts"]["stages"].values(),
        manifest["artifacts"]["preview"]["video"],
        manifest["artifacts"]["preview"]["thumbnail"],
    ]:
        assert digest(run / item["path"]) == item["sha256"]
    reference = load_rimkit_reference(run / "final/robot_motion.npz")
    if not reference.scientific_source_ready:
        raise ValueError("CATO scientific source gate: " + str(reference.source_gate))
    names = kinematic_joint_names_from_mjcf(
        CATO / "resources/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml"
    )
    motion = reference.mimickit_motion(names)
    native = directory / "motion.pkl"
    tmp = native.with_suffix(".tmp.pkl")
    with tmp.open("wb") as stream:
        pickle.dump(
            {
                "loop_mode": 0,
                "fps": motion.fps,
                "frames": motion.frames.astype(np.float32).tolist(),
            },
            stream,
            protocol=4,
        )
    os.replace(tmp, native)
    sampled = directory / "motion_50hz.pkl"
    legacy_export = export_optional_50hz(motion, sampled)
    restored = load_mimickit_motion(native, names)
    assert restored.frames.shape == (len(reference.arrays["timestamps_s"]), 29)
    assert restored.fps == 30.0 and restored.loop_mode == 0
    assert np.allclose(restored.frames, motion.frames, rtol=0, atol=5e-7)
    pins = [
        source,
        native,
        run / "manifest.json",
        run / "final/robot_motion.npz",
        run / "stages/7_fpa_ik.npz",
        run / "stages/8_final.npz",
        run / "preview/final.mp4",
        run / "preview/final.png",
    ]
    if sampled.exists():
        pins.append(sampled)
    record = {
        "schema_version": 1,
        "motion_id": "bones_" + row["motion_group"],
        "motion_group": row["motion_group"],
        "selection_filename": row["filename"],
        "source_bvh": row["uniform_bvh_path"],
        "source_bvh_sha256": metadata["source_sha256"],
        "skeleton": "SOMA Uniform",
        "robot": "k1",
        "fps": motion.fps,
        "frame_count": len(motion.frames),
        "duration_seconds": reference.duration,
        "alpha_dimension": reference.alpha_dimension,
        "boundaries": list(reference.boundaries),
        "segment_frames": reference.arrays["cato_segment_frames"].tolist(),
        "joint_names": list(names),
        "source_modification": (
            "30Hz resampling and floor alignment; no trim, padding, or ready transition"
        ),
        "scientific_source_ready": True,
        "legacy_50hz_export": legacy_export,
        "pipeline_warnings": manifest.get("warnings", []),
        "compute_backend": manifest["compute_backend"],
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "file_pins": {str(p.relative_to(root)): digest(p) for p in pins},
    }
    atomic_json(directory / "result.json", record)
    return record


def worker(args, row):
    root = args.output
    if complete(root, row):
        return {"motion_group": row["motion_group"], "status": "complete", "resumed": True}
    log = root / "logs" / f'{row["motion_group"]}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="egl",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join([str(RIMKIT / "src"), str(CATO / "src")]),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--selection",
        str(args.selection),
        "--output",
        str(root),
        "--one",
        row["motion_group"],
        "--backend",
        args.backend,
    ]
    with log.open("a") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=RIMKIT
        )
    return {
        "motion_group": row["motion_group"],
        "status": "complete" if result.returncode == 0 and complete(root, row) else "failed",
        "returncode": result.returncode,
        "log": str(log),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=["native", "python", "auto"], default="native")
    parser.add_argument("--one")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--status-name", default="batch_status.json")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.selection.read_text())
    if args.one:
        row = next(x for x in rows if x["motion_group"] == args.one)
        try:
            print(
                json.dumps(one(row, args.output, args.backend), ensure_ascii=False, indent=2),
                flush=True,
            )
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    if args.limit:
        rows = rows[: args.limit]
    if args.reverse:
        rows.reverse()
    results = []
    started = time.monotonic()
    atomic_json(args.output / "selection.json", json.loads(args.selection.read_text()))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, args, row): row for row in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "motion_group": futures[future]["motion_group"],
                    "status": "failed",
                    "error": repr(error),
                }
            results.append(result)
            state = {
                "total": len(rows),
                "finished": len(results),
                "complete": sum(x["status"] == "complete" for x in results),
                "failed": sum(x["status"] == "failed" for x in results),
                "elapsed_seconds": time.monotonic() - started,
                "results": results,
            }
            atomic_json(args.output / args.status_name, state)
            print(
                json.dumps({k: v for k, v in state.items() if k != "results"}),
                result["motion_group"],
                result["status"],
                flush=True,
            )
    return 0 if all(x["status"] == "complete" for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
