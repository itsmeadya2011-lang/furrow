from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from furrow.config import Plan


class SessionState(BaseModel):
    goal: str
    cycles: int = 0
    current_plan: Plan | None = None
    updated_at: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def save(self, path: Path) -> None:
        self.updated_at = datetime.utcnow().isoformat() + "Z"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        if self.current_plan is not None:
            data["current_plan"] = self.current_plan.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> SessionState | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            plan_data = raw.pop("current_plan", None)
            state = cls(**raw)
            if plan_data is not None:
                state.current_plan = Plan(**plan_data)
            return state
        except (json.JSONDecodeError, ValueError):
            return None
