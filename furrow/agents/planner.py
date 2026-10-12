from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient, strip_code_fences

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(strip_code_fences(response))
        except json.JSONDecodeError as e:
            retry_prompt = (
                f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\n"
                "Return only valid JSON for the plan. No markdown fences, no explanation."
            )
            response = await self.client.complete(retry_prompt, model=self.client.settings.planner_model)
            try:
                data = json.loads(strip_code_fences(response))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

        if not data.get("tasks"):
            raise ValueError("Planner returned no tasks")

        seen_ids: set[str] = set()
        for task in data["tasks"]:
            task_id = task.get("id")
            if task_id in seen_ids:
                raise ValueError(f"Planner returned duplicate task ids: {task_id}")
            seen_ids.add(task_id)

        return Plan(**data)