from __future__ import annotations

import asyncio
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
        last_error = None
        last_response = ""
        for attempt in range(3):
            try:
                prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
                response = await self.client.complete(prompt, model=self.client.settings.planner_model)
                last_response = response
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(1)
        raise ValueError(f"Failed to parse plan after 3 attempts: {last_error}\nLast response: {last_response}")
