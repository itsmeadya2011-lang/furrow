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

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None, task_results: dict[str, str] | None = None) -> Plan:
        existing_tasks = existing_tasks or []
        task_results = task_results or {}
        
        # Build context from previous tasks
        context = ""
        if existing_tasks:
            context += "\nPrevious tasks:\n"
            for task in existing_tasks:
                status = task.status
                result = task_results.get(task.id, task.result or "no result")
                context += f"  - {task.id}: {task.description} [{status}]\n"
                if status == "completed":
                    context += f"    Result: {result[:200]}...\n" if len(result) > 200 else f"    Result: {result}\n"
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")