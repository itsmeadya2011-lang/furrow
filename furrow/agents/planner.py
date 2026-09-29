from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, max_retries: int = 3) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_error: Exception | None = None
        for attempt in range(max_retries):
            response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                # Try again with an explicit reminder.
                prompt = (
                    f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
                    f"\nPrevious response was not valid JSON. "
                    f"Return ONLY a JSON object matching the schema.\n"
                )
        raise ValueError(
            f"Failed to parse plan after {max_retries} attempts: {last_error}"
        )
