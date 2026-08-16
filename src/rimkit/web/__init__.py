"""Local browser interface for RIMKit's current CoRe method.

FastAPI is an optional dependency. Import :func:`create_app` from
``rimkit.web.app`` only after installing the ``web`` extra.
"""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create the optional FastAPI application without eager web imports."""

    from rimkit.web.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
