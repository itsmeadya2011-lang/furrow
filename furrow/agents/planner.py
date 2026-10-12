from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.util import extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = extract_json(response)
            data.setdefault("rationale", "")
            normalized_tasks: list[dict[str, Any]] = []
            for index, task in enumerate(data.get("tasks", [])):
                if not isinstance(task, dict):
                    continue
                task = dict(task)
                if not task.get("id"):
                    task["id"] = str(index + 1)
                task.setdefault("files", [])
                task.setdefault("dependencies", [])
                normalized_tasks.append(task)
            data["tasks"] = normalized_tasks
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e