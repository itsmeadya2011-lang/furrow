"""Core loop: event bus and orchestrator.

This package owns the run loop and the progress-event contract that the web and CLI
front-ends build on. Keeping it isolated here means ``furrow.web`` and ``furrow.cli`` can
import ``Orchestrator`` and the event bus without pulling in provider SDKs transitively.
"""

from furrow.core.events import Event, EventEmitter, EventType
from furrow.core.orchestrator import (
    DEFAULT_NO_PROGRESS_LIMIT,
    Orchestrator,
    RunSummary,
)

__all__ = [
    "DEFAULT_NO_PROGRESS_LIMIT",
    "Event",
    "EventEmitter",
    "EventType",
    "Orchestrator",
    "RunSummary",
]
