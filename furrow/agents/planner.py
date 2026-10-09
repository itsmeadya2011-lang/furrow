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
        self.client = client or LLMClient(settings=settings or Settings())

    async def plan(self, goal: str, existing_tasks: dict[str, TaskModel] | None = None) -> Plan:
        existing = ""
        if existing_tasks:
            completed = [f"- {t.id}: {t.description} (completed)" for t in existing_tasks.values() if t.status == "completed"]
            failed = [f"- {t.id}: {t.description} (failed: {t.result})" for t in existing_tasks.values() if t.status == "failed"]
            if completed or failed:
                existing = "\n\nExisting task history:\n" + "\n".join(completed + failed)

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{existing}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
