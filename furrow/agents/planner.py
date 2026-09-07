from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_response = ""
        for attempt in range(3):
            if attempt > 0:
                prompt = f"{prompt}\nReturn only valid JSON, no explanation."
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            last_response = response
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Planner attempt %d failed to parse JSON response", attempt + 1)
        raise ValueError(f"Failed to parse plan from LLM after 3 attempts\nLast response: {last_response}")
