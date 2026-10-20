from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """A progress event emitted by the Orchestrator during a cycle."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.data}
