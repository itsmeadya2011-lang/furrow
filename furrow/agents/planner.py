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

    async def plan(self, goal: str, previous_plans: list[Plan] | None = None, all_tasks: list[TaskModel] | None = None) -> Plan:
        context = ""
        if previous_plans:
            context += "\n\nPrevious plans:\n"
            for i, p in enumerate(previous_plans):
                context += f"Cycle {i+1}: {p.rationale}\n"
                for t in p.tasks:
                    context += f"  - {t.id}: {t.description} [{t.status}]\n"
        
        if all_tasks:
            completed = [t for t in all_tasks if t.status == "completed"]
            failed = [t for t in all_tasks if t.status == "failed"]
            if completed:
                context += f"\n\nCompleted tasks ({len(completed)}):\n"
                for t in completed:
                    context += f"  - {t.id}: {t.description}\n"
            if failed:
                context += f"\n\nFailed tasks ({len(failed)}):\n"
                for t in failed:
                    context += f"  - {t.id}: {t.description} - {t.result}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
