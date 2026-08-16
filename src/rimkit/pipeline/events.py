"""Interface-neutral progress events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from rimkit.pipeline.state import PipelineStage


class PipelineEventType(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    PROGRESS = "progress"
    METRIC = "metric"
    WARNING = "warning"
    PREVIEW = "preview"
    ARTIFACT = "artifact"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass(frozen=True)
class PipelineEvent:
    event_type: PipelineEventType
    stage: PipelineStage
    job_id: str
    robot_id: str
    message: str = ""
    current: int | None = None
    total: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        payload["stage"] = self.stage.value
        return payload


class EventSink(Protocol):
    def emit(self, event: PipelineEvent) -> None:
        """Receive one pipeline event."""


class NullEventSink:
    def emit(self, event: PipelineEvent) -> None:
        del event


class CallbackEventSink:
    def __init__(self, callback: Callable[[PipelineEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: PipelineEvent) -> None:
        self._callback(event)
