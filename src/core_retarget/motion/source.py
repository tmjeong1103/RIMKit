"""Container-aware SOMA source dispatch for Kimodo NPZ and GEM-X PT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core_retarget.exceptions import MotionValidationError
from core_retarget.motion.contacts import ContactSchedule, build_contact_schedule
from core_retarget.motion.gemx import GemxContactSeed, load_gemx_motion
from core_retarget.motion.soma import (
    SomaMotion,
    SomaMotionSummary,
    load_soma_motion,
    validate_soma_npz,
)

SourceContainer = Literal["npz", "pt"]
SourceProvider = Literal["kimodo", "gem-x"]


@dataclass(frozen=True, slots=True)
class SourceMotionSummary:
    """Provider-aware metadata shared by CLI, API, web, and manifests."""

    path: Path
    sha256: str
    frame_count: int
    fps: float
    duration_seconds: float
    keys: tuple[str, ...]
    contact_channels: int | None
    warnings: tuple[str, ...]
    container_format: SourceContainer
    provider: SourceProvider


@dataclass(frozen=True, slots=True)
class LoadedSourceMotion:
    """A source container normalized to the common SOMA77 in-memory contract."""

    summary: SourceMotionSummary
    motion: SomaMotion
    gemx_contacts: GemxContactSeed | None = None

    def __post_init__(self) -> None:
        if self.motion.path != self.summary.path or self.motion.sha256 != self.summary.sha256:
            raise MotionValidationError("Loaded source metadata does not match its SOMA motion.")
        if (
            self.motion.frame_count != self.summary.frame_count
            or self.motion.fps != self.summary.fps
        ):
            raise MotionValidationError("Loaded source timeline does not match its metadata.")
        if (self.summary.provider == "gem-x") != (self.gemx_contacts is not None):
            raise MotionValidationError("GEM-X contact metadata is inconsistent with the provider.")

    def build_contact_schedule(self) -> ContactSchedule:
        """Build the provider-specific contact schedule used by all stages."""

        if self.gemx_contacts is None:
            return build_contact_schedule(self.motion)
        return build_contact_schedule(
            self.motion,
            left_source_labels=self.gemx_contacts.left_fused,
            right_source_labels=self.gemx_contacts.right_fused,
            source_contact_name="gemx_fused_toebase_contacts+time_varying_floor",
            floor_distance_threshold=0.03,
            maximum_contact_bridge_time=0.30,
        )


def _summary(
    summary: SomaMotionSummary,
    *,
    container_format: SourceContainer,
    provider: SourceProvider,
) -> SourceMotionSummary:
    return SourceMotionSummary(
        path=summary.path,
        sha256=summary.sha256,
        frame_count=summary.frame_count,
        fps=summary.fps,
        duration_seconds=summary.duration_seconds,
        keys=summary.keys,
        contact_channels=summary.contact_channels,
        warnings=summary.warnings,
        container_format=container_format,
        provider=provider,
    )


def _container(path: str | Path) -> SourceContainer:
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".pt":
        return "pt"
    raise MotionValidationError(
        f"Unsupported source motion extension {suffix or '<none>'!r}; expected .npz or .pt."
    )


def validate_source_motion(
    path: str | Path,
    *,
    fps_override: float | None = None,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
    max_frames: int = 1_000_000,
) -> SourceMotionSummary:
    """Validate either a Kimodo NPZ or GEM-X PT based on its extension."""

    container = _container(path)
    if container == "npz":
        validated = validate_soma_npz(
            path,
            fps_override=fps_override,
            max_file_bytes=max_file_bytes,
            max_frames=max_frames,
        )
        return _summary(validated, container_format="npz", provider="kimodo")
    motion, _ = load_gemx_motion(
        path,
        fps_override=fps_override,
        max_file_bytes=max_file_bytes,
        max_frames=max_frames,
    )
    return _summary(motion.summary, container_format="pt", provider="gem-x")


def load_source_motion(
    path: str | Path,
    *,
    fps_override: float | None = None,
    max_file_bytes: int = 2 * 1024 * 1024 * 1024,
    max_frames: int = 1_000_000,
) -> LoadedSourceMotion:
    """Load and normalize a Kimodo NPZ or GEM-X PT source motion."""

    container = _container(path)
    if container == "npz":
        motion_path = Path(path).expanduser().resolve()
        if not motion_path.is_file():
            raise MotionValidationError(f"Motion file does not exist: {motion_path}")
        if motion_path.stat().st_size > max_file_bytes:
            raise MotionValidationError(
                f"Motion file is larger than the {max_file_bytes}-byte safety limit."
            )
        motion = load_soma_motion(motion_path, fps_override=fps_override)
        if motion.frame_count > max_frames:
            raise MotionValidationError(
                f"Motion has {motion.frame_count} frames, exceeding the limit of {max_frames}."
            )
        return LoadedSourceMotion(
            summary=_summary(motion.summary, container_format="npz", provider="kimodo"),
            motion=motion,
        )
    motion, contacts = load_gemx_motion(
        path,
        fps_override=fps_override,
        max_file_bytes=max_file_bytes,
        max_frames=max_frames,
    )
    return LoadedSourceMotion(
        summary=_summary(motion.summary, container_format="pt", provider="gem-x"),
        motion=motion,
        gemx_contacts=contacts,
    )


__all__ = [
    "LoadedSourceMotion",
    "SourceContainer",
    "SourceMotionSummary",
    "SourceProvider",
    "load_source_motion",
    "validate_source_motion",
]
