from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel, TaskStatus
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, incomplete_tasks: list[TaskModel] | None = None) -> Plan:
        incomplete_tasks = incomplete_tasks or []
        
        # Build context from incomplete tasks
        context = ""
        if incomplete_tasks:
            context = "\n\nIncomplete tasks from previous cycles:\n"
            for task in incomplete_tasks:
                context += f"- Task {task.id} ({task.status.value}): {task.description}\n"
                if task.result:
                    context += f"  Result: {task.result[:200]}\n"
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
