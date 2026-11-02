#!/usr/bin/env python3
"""Convert one BONES-SEED SOMA BVH for RIMKit.

SOMA Uniform (``soma_uniform/bvh``) is the recommended source. SOMA Proportional
BVH is accepted for explicit comparisons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rimkit.motion.bvh import convert_bones_seed_bvh
from rimkit.motion.soma import validate_soma_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BONES-SEED soma_uniform/bvh/<date>/<name>.bvh")
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--target-fps", type=float, help="Resample while preserving playback speed."
    )
    parser.add_argument(
        "--ground", action="store_true", help="Subtract the 1st-percentile toe floor."
    )
    parser.add_argument(
        "--estimate-contacts",
        action="store_true",
        help="Store source kinematic contact estimates and their provenance.",
    )
    args = parser.parse_args()
    if "soma_uniform" not in args.source.expanduser().resolve().parts:
        print(
            "warning: SOMA Uniform (soma_uniform/bvh) is the recommended BONES-SEED source.",
            file=sys.stderr,
        )
    output = convert_bones_seed_bvh(
        args.source,
        args.destination,
        target_fps=args.target_fps,
        ground=args.ground,
        estimate_contacts=args.estimate_contacts,
    )
    summary = validate_soma_npz(output)
    print(
        f"Saved {output}: {summary.frame_count} frames at {summary.fps:g} Hz "
        f"({summary.duration_seconds:.3f} s), Y-up meters, SOMA77."
    )


if __name__ == "__main__":
    main()
