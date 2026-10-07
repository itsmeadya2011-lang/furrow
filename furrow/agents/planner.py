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
        completed = completed_tasks or []
        completed_summary = ""
        if completed:
            completed_summary = "\n\nAlready completed tasks:\n" + "\n".join(
                f"- {t.id}: {t.description} ({t.status})" for t in completed
            )
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{completed_summary}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
