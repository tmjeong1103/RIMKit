"""Packaged MuJoCo scenes and model assets."""

from pathlib import Path


def root_path() -> Path:
    """Return the installed asset root without relying on the current directory."""

    return Path(__file__).resolve().parent


__all__ = ["root_path"]
