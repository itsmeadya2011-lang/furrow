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

    async def plan(self, goal: str, previous_tasks: list[TaskModel] | None = None) -> Plan:
        prev_context = ""
        if previous_tasks:
            completed = [t for t in previous_tasks if t.status == "completed"]
            failed = [t for t in previous_tasks if t.status == "failed"]
            pending = [t for t in previous_tasks if t.status == "pending"]
            prev_context = f"\n\nPrevious tasks:\n"
            if completed:
                prev_context += f"Completed: {', '.join(f'{t.id}({t.description[:30]})' for t in completed)}\n"
            if failed:
                prev_context += f"Failed: {', '.join(f'{t.id}({t.description[:30]})' for t in failed)}\n"
            if pending:
                prev_context += f"Pending: {', '.join(f'{t.id}({t.description[:30]})' for t in pending)}\n"

        prompt = f"{PLANNER_PROMPT}{prev_context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
