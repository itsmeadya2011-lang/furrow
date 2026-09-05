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
        previous_tasks: list[TaskModel] | None = None,
        history: list[dict] | None = None,
    ) -> Plan:
        context = ""
        if previous_tasks:
            completed = [t for t in previous_tasks if t.status == "completed"]
            failed = [t for t in previous_tasks if t.status == "failed"]
            if completed:
                context += f"\nCompleted tasks ({len(completed)}):\n"
                for t in completed[:5]:
                    result_preview = (t.result or "")[:200]
                    context += f"  - {t.id}: {result_preview}\n"
            if failed:
                context += f"\nFailed tasks ({len(failed)}):\n"
                for t in failed:
                    context += f"  - {t.id}: {t.result}\n"
        if history:
            context += f"\nPrevious cycles: {len(history)}\n"
            for h in history[-3:]:
                status = "passed" if h["test_result"]["passed"] else "failed"
                context += f"  Cycle {h['cycle']}: tests {status}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
