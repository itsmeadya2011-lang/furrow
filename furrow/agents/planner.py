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

    async def plan(
        self,
        goal: str,
        existing_tasks: list[TaskModel] | None = None,
        cycle_history: list[dict] | None = None,
    ) -> Plan:
        history_str = "No previous cycles."
        if cycle_history:
            history_str = "\n".join(
                f"Cycle {h['cycle']}: {h['plan'].get('rationale', '')} - {len(h['tasks'])} tasks"
                for h in cycle_history[-3:]
            )

        tasks_str = "No existing tasks."
        if existing_tasks:
            tasks_str = "\n".join(f"- {t.id}: {t.description} ({t.status})" for t in existing_tasks)

        prompt = PLANNER_PROMPT.format(history=history_str, tasks=tasks_str) + f"\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
