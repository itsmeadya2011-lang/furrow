"""Observable lifecycle events emitted by the :mod:`furrow.core` orchestrator.

The orchestrator runs unattended for an unbounded amount of time, so its only
useful interface for anything other than a terminal is a stream of events. This
module defines that stream: :class:`EventType` names every step the loop can
take, :class:`FurrowEvent` carries one such step, and :class:`RunResult`
summarises the run once it halts.

Every model is JSON-serialisable with ``model_dump(mode="json")``, which is the
form the web server forwards to browser clients:

Example:
    ```python
    from furrow.core import EventType, FurrowEvent

    event = FurrowEvent(type=EventType.CYCLE_START, message="Cycle 1", cycle=1)
    assert event.model_dump(mode="json")["type"] == "cycle_start"
    ```
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Every point in a run at which the orchestrator reports progress.

    The values are stable wire strings: consumers such as the web server switch
    on them, so they must not be renamed without updating those consumers.
    """

    RUN_START = "run_start"
    CYCLE_START = "cycle_start"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    TEST_START = "test_start"
    TEST_RESULT = "test_result"
    CYCLE_END = "cycle_end"
    ERROR = "error"
    RUN_END = "run_end"


class FurrowEvent(BaseModel):
    """One observation made by the orchestrator while a run is in flight.

    Attributes:
        type: Which lifecycle step this observation describes.
        message: Human-readable summary, safe to show as-is in a log or UI.
        cycle: One-based cycle number the event belongs to; ``0`` for events
            emitted outside a cycle, such as :attr:`EventType.RUN_START`.
        task_id: Identifier of the task the event concerns, when applicable.
        data: Structured payload whose keys depend on ``type``. Always
            JSON-serialisable via ``model_dump(mode="json")``.
    """

    type: EventType
    message: str = ""
    cycle: int = 0
    task_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Summary of a finished run, returned by ``Orchestrator.run``.

    Attributes:
        cycles: Number of plan/execute/test cycles that were started.
        completed: Whether the loop reached its goal, as opposed to halting
            because of a limit, a stall, or an unrecoverable error.
        tasks_completed: Tasks in the registry left in ``COMPLETED`` state.
        tasks_failed: Tasks in the registry left in ``FAILED`` state.
        cycles_failed: Cycles that aborted with an exception before finishing.
        last_error: Message of the most recent error seen, or ``None`` if no
            cycle has failed. Sticky: it is not cleared by a later good cycle.
        stopped_reason: Why the loop halted, always populated by ``run()``.
    """

    cycles: int = 0
    completed: bool = False
    tasks_completed: int = 0
    tasks_failed: int = 0
    cycles_failed: int = 0
    last_error: str | None = None
    stopped_reason: str = ""
