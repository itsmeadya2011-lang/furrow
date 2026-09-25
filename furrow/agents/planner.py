from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing = existing_tasks or []
        completed = [t for t in existing if t.status == "completed"]
        pending = [t for t in existing if t.status in ("pending", "failed")]

        context = ""
        if completed:
            context += "\nCompleted tasks:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in completed)
        if pending:
            context += "\nPending/failed tasks:\n" + "\n".join(f"  - {t.id}: {t.description} ({t.status})" for t in pending)

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        if existing:
            prompt += "\nOnly plan NEW tasks that haven't been done yet. Don't duplicate completed work."
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
