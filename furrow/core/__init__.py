"""Core loop of Furrow: the orchestrator and the events it publishes.

Example:
    ```python
    from furrow.core import EventType, Orchestrator, RunResult

    events: list = []
    result: RunResult = await Orchestrator("fix the parser", on_event=events.append).run()
    ```
"""

from furrow.core.events import EventType, FurrowEvent, RunResult
from furrow.core.orchestrator import Orchestrator

__all__ = ["EventType", "FurrowEvent", "Orchestrator", "RunResult"]
