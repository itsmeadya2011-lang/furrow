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
        self.settings = settings

    async def plan(
        self,
        goal: str,
        completed_tasks: list[TaskModel] | None = None,
        cycle_history: list[dict] | None = None,
    ) -> Plan:
        completed_tasks = completed_tasks or []
        cycle_history = cycle_history or []

        # Build context from previous cycles
        context_parts = []

        if completed_tasks:
            completed_list = [t for t in completed_tasks if t.status == "completed"]
            failed_list = [t for t in completed_tasks if t.status == "failed"]
            pending_list = [t for t in completed_tasks if t.status == "pending"]

            if completed_list:
                context_parts.append(
                    "## Completed Tasks:\n"
                    + "\n".join(f"- {t.id}: {t.description} → {t.result[:100] if t.result else 'done'}" for t in completed_list)
                )
            if failed_list:
                context_parts.append(
                    "## Failed Tasks (need fixing):\n"
                    + "\n".join(f"- {t.id}: {t.description} → {t.result[:100] if t.result else 'failed'}" for t in failed_list)
                )
            if pending_list:
                context_parts.append(
                    "## Pending Tasks:\n"
                    + "\n".join(f"- {t.id}: {t.description}" for t in pending_list)
                )

        if cycle_history:
            recent = cycle_history[-3:]  # Last 3 cycles
            context_parts.append(
                "## Recent Cycles:\n"
                + "\n".join(
                    f"Cycle {c['cycle']}: {len(c.get('tasks', []))} tasks, "
                    f"tests={'passed' if c.get('test_result', {}).get('passed') else 'failed'}"
                    for c in recent
                )
            )

        context = "\n\n".join(context_parts) if context_parts else "No previous context."

        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")