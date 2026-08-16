#!/usr/bin/env python3
"""Generate or verify deterministic SHA-256 manifests for robot assets."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ASSETS = REPOSITORY_ROOT / "src" / "rimkit" / "assets" / "robots"
VENDORS = (
    "unitree",
    "robotis",
    "apptronik",
    "limx",
    "fourier",
    "pndbotics",
    "booster",
    "engineai",
)


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest_text(vendor: str) -> str:
    root = ROBOT_ASSETS / vendor
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "FILES.sha256":
            continue
        lines.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def write_manifest(vendor: str) -> None:
    path = ROBOT_ASSETS / vendor / "FILES.sha256"
    path.write_text(manifest_text(vendor), encoding="utf-8")
    print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")


def check_manifest(vendor: str) -> bool:
    path = ROBOT_ASSETS / vendor / "FILES.sha256"
    if not path.is_file():
        print(f"missing {path.relative_to(REPOSITORY_ROOT)}")
        return False
    expected = path.read_text(encoding="utf-8")
    actual = manifest_text(vendor)
    if expected != actual:
        print(f"stale {path.relative_to(REPOSITORY_ROOT)}")
        return False
    print(f"ok {path.relative_to(REPOSITORY_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--vendor", choices=(*VENDORS, "all"), default="all")
    args = parser.parse_args()
    vendors = VENDORS if args.vendor == "all" else (args.vendor,)
    if args.action == "write":
        for vendor in vendors:
            write_manifest(vendor)
        return 0
    return 0 if all(check_manifest(vendor) for vendor in vendors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
