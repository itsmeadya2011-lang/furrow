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

    async def plan(self, goal: str, completed_tasks: list[TaskModel] | None = None, failed_tasks: list[TaskModel] | None = None) -> Plan:
        context = ""
        if completed_tasks:
            context += "\nCompleted tasks:\n"
            for task in completed_tasks:
                context += f"  - {task.id}: {task.description} (result: {task.result[:100] if task.result else 'N/A'}...)\n"
        if failed_tasks:
            context += "\nFailed tasks:\n"
            for task in failed_tasks:
                context += f"  - {task.id}: {task.description} (error: {task.result[:100] if task.result else 'N/A'}...)\n"
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
