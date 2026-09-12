from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(
        self,
        goal: str,
        completed_tasks: list[str] | None = None,
        failed_tasks: list[str] | None = None,
        last_test_failures: list[str] | None = None,
        cycle_number: int | None = None,
    ) -> Plan:
        context_parts = []
        if cycle_number is not None:
            context_parts.append(f"Current cycle number: {cycle_number}")
        if completed_tasks:
            context_parts.append(
                "Already completed tasks (do not re-plan these):\n"
                + "\n".join(f"• {t}" for t in completed_tasks)
            )
        if failed_tasks:
            context_parts.append(
                "Previously failed tasks (focus on fixing these):\n"
                + "\n".join(f"• {t}" for t in failed_tasks)
            )
        if last_test_failures:
            context_parts.append(
                "Test failures to address:\n"
                + "\n".join(f"• {f}" for f in last_test_failures)
            )

        context_block = "\n\n".join(context_parts)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if context_block:
            prompt += f"\n\nPrevious cycle context:\n{context_block}\n"

        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
