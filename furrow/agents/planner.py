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

    async def plan(self, goal: str, history: list[TaskModel] | None = None) -> Plan:
        history = history or []
        history_str = ""
        if history:
            history_str = "\n\nPrevious tasks (completed/failed):\n"
            for task in history:
                history_str += f"- {task.id}: {task.description} [{task.status}]\n"
                if task.result:
                    history_str += f"  Result: {task.result[:200]}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{history_str}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")