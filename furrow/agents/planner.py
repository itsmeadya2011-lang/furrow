from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, completed_tasks: list[TaskModel] | None = None) -> Plan:
        completed_info = ""
        if completed_tasks:
            completed_info = "\n\nAlready completed tasks:\n"
            for task in completed_tasks:
                completed_info += f"  - {task.id}: {task.description} (status: {task.status})\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{completed_info}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
