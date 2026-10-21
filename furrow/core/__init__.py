from typing import TYPE_CHECKING, Any

from furrow.core.events import Event, EventBus, EventType
from furrow.core.state import RunState, StateStore

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator

__all__ = ["Event", "EventBus", "EventType", "Orchestrator", "RunState", "StateStore"]


def __getattr__(name: str) -> Any:
    """Import the orchestrator lazily.

    The orchestrator imports the agents, and the agents import helpers from
    ``furrow.core`` (e.g. ``furrow.core.proc``).  Importing it eagerly here
    would close an import cycle, so it is resolved on first use instead.
    """
    if name == "Orchestrator":
        from furrow.core.orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
