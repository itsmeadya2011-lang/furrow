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

    async def plan(self, goal: str, max_attempts: int = 3) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_error: Exception | None = None
        for _ in range(max_attempts):
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                # Steer the model toward valid JSON on the next attempt
                prompt = (
                    f"{PLANNER_PROMPT}\n\n"
                    f"Goal: {goal}\n"
                    f"The previous response was not valid JSON. "
                    f"Return ONLY valid JSON matching the schema, no markdown.\n"
                    f"Previous response was: {response[:500]}\n"
                )
        raise ValueError(
            f"Failed to parse plan from LLM after {max_attempts} attempts: {last_error}"
        )
