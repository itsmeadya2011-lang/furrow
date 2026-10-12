from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        data = await self.client.complete_json(prompt, model=self.client.settings.planner_model)

        if not isinstance(data, dict):
            raise ValueError("Planner response is not a JSON object")

        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Planner returned no tasks")

        # Clamp to at most 5 tasks and normalize each entry.
        normalized: list[dict] = []
        for index, task in enumerate(tasks[:5], start=1):
            if not isinstance(task, dict):
                continue
            task.setdefault("id", str(index))
            task.setdefault("files", [])
            task.setdefault("dependencies", [])
            normalized.append(task)

        if not normalized:
            raise ValueError("Planner returned no tasks")

        data["tasks"] = normalized
        data.setdefault("rationale", "")
        return Plan(**data)