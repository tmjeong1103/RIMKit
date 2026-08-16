"""Registry of retargeting methods exposed by RIMKit."""

from __future__ import annotations

from dataclasses import dataclass

from rimkit.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Stable public metadata for an available retargeting method."""

    method_id: str
    display_name: str
    description: str


_METHODS = (
    MethodSpec(
        method_id="core",
        display_name="CoRe",
        description="Contact-aware whole-body motion retargeting for humanoid robots.",
    ),
)


def list_methods() -> tuple[MethodSpec, ...]:
    """Return all methods in stable CLI and documentation order."""

    return _METHODS


def get_method(method_id: str) -> MethodSpec:
    """Return one method by its case-insensitive public identifier."""

    normalized = method_id.strip().lower()
    for method in _METHODS:
        if method.method_id == normalized:
            return method
    choices = ", ".join(method.method_id for method in _METHODS)
    raise ConfigurationError(f"Unknown method {method_id!r}; choose one of: {choices}.")


__all__ = ["MethodSpec", "get_method", "list_methods"]
