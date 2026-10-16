from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(
        self,
        goal: str,
        previous_tasks: list[TaskModel] | None = None,
    ) -> Plan:
        context = ""
        if previous_tasks:
            completed = [t for t in previous_tasks if t.status == "completed"]
            failed = [t for t in previous_tasks if t.status in ("failed", "blocked")]
            if completed:
                context += "\nPreviously completed tasks:\n"
                for t in completed:
                    context += f"- {t.id}: {t.description}\n  Result: {t.result}\n"
            if failed:
                context += "\nPreviously failed or blocked tasks:\n"
                for t in failed:
                    context += f"- {t.id}: {t.description}\n  Error: {t.result}\n"
            context += (
                "\nDo not re-plan completed tasks. Focus on remaining work or fixing failures.\n"
            )

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
