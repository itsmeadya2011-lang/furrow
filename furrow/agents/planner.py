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
        context = ""
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            if completed:
                context += "\nCompleted tasks:\n"
                for t in completed:
                    context += f"  - {t.id}: {t.description} -> {t.result[:100]}\n"
            if failed:
                context += "\nFailed tasks:\n"
                for t in failed:
                    context += f"  - {t.id}: {t.description} -> {t.result}\n"
        
        if cycle_history:
            context += f"\nPrevious cycles: {len(cycle_history)}\n"
            last_cycle = cycle_history[-1] if cycle_history else None
            if last_cycle:
                test_result = last_cycle.get("test_result", {})
                if test_result.get("failures"):
                    context += f"Last test failures: {test_result['failures']}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{context}\nReturn only new tasks needed."
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
