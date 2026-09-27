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

    async def plan(self, goal: str) -> Plan:
        prompt = f"Goal: {goal}\n"
        last_error: Exception | None = None
        for attempt in range(3):
            if last_error is not None:
                prompt = (
                    f"Your previous response was not valid JSON.\n"
                    f"Error: {last_error}\n"
                    f"Please retry and output only valid JSON matching the schema. "
                    f"Goal: {goal}\n"
                )
            response = await self.client.complete(
                prompt,
                system=PLANNER_PROMPT,
                model=self.client.settings.planner_model,
            )
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
        raise ValueError(
            f"Failed to parse plan from LLM after 3 attempts: {last_error}\n"
            f"Last response: {response}"
        )
