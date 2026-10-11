from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing_tasks = existing_tasks or []

        # Build context about existing tasks
        context_parts = []
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            pending = [t for t in existing_tasks if t.status == "pending"]

            if completed:
                context_parts.append("Completed tasks:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in completed))
            if failed:
                context_parts.append("Failed tasks (need fix):\n" + "\n".join(f"  - {t.id}: {t.description} - {t.result}" for t in failed))
            if pending:
                context_parts.append("Pending tasks:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in pending))

        context = "\n\n".join(context_parts) if context_parts else "No previous tasks."

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nContext:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
