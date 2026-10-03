from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Every kind of progress notification emitted by the orchestrator."""

    CYCLE_START = "cycle_start"
    PLAN = "plan"
    TASK_START = "task_start"
    TASK_END = "task_end"
    TEST_START = "test_start"
    TEST_END = "test_end"
    CYCLE_END = "cycle_end"
    ERROR = "error"
    DONE = "done"


@dataclass
class Event:

    type: EventType
    cycle: int = 0
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the event."""
        return {
            "type": self.type.value,
            "cycle": self.cycle,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }