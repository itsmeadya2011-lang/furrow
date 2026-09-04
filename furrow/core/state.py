from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from furrow.config import Plan


class StateStore:
    """Persist orchestrator state (goals, plans, cycle history) to a JSON file.

    The store is append-only on each cycle so progress survives crashes and
    session restarts. ``load_latest()`` returns the most recent state record
    so a new orchestrator can resume from where the previous one stopped.
    """

    def __init__(self, path: str | Path = ".furrow/state.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """Append a JSON-encoded record (one per line)."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def load_latest(self) -> Optional[dict]:
        """Return the most recently appended record, or None if the file is empty."""
        if not self.path.exists():
            return None
        last: Optional[dict] = None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        return last

    def save_plan(self, goal: str, cycle: int, plan: "Plan") -> None:
        self.append(
            {
                "type": "plan",
                "goal": goal,
                "cycle": cycle,
                "rationale": plan.rationale,
                "tasks": [t.model_dump() for t in plan.tasks],
            }
        )

    def save_cycle_result(self, goal: str, cycle: int, passed: bool, summary: str) -> None:
        self.append(
            {
                "type": "cycle_result",
                "goal": goal,
                "cycle": cycle,
                "passed": passed,
                "summary": summary,
            }
        )