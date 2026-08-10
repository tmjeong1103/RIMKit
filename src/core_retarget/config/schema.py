"""Public options that are applied by the currently executable stages."""

from __future__ import annotations

from dataclasses import dataclass

from core_retarget.exceptions import ConfigurationError
from core_retarget.native import BackendPreference
from core_retarget.robots.registry import get_robot


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Robot, optional source-FPS override, and compute implementation.

    ``backend`` selects the faithful native or Python implementation; it is not
    a solver-quality preset and does not change retargeting parameters.
    """

    robot: str
    fps: float | None = None
    backend: BackendPreference = "auto"

    def __post_init__(self) -> None:
        get_robot(self.robot)
        if self.fps is not None and not 0.0 < self.fps <= 1000.0:
            raise ConfigurationError("FPS must be in (0, 1000].")
        if self.backend not in {"auto", "native", "python"}:
            raise ConfigurationError(
                "Compute backend must be 'auto', 'native', or 'python'."
            )
