from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing = existing_tasks or []
        completed = [t for t in existing if t.status == "completed"]
        pending = [t for t in existing if t.status != "completed"]

        context = ""
        if completed:
            context += f"\nAlready completed tasks:\n"
            for t in completed:
                context += f"  - {t.id}: {t.description} (result: {t.result[:100] if t.result else 'none'})\n"
        if pending:
            context += f"\nPending/In-progress tasks:\n"
            for t in pending:
                context += f"  - {t.id}: {t.description} (status: {t.status})\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")