from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_JSON_RETRIES = 2


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)

        last_error: Exception | None = None
        for attempt in range(MAX_JSON_RETRIES + 1):
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < MAX_JSON_RETRIES:
                    retry_prompt = (
                        "Your previous response was not valid JSON "
                        f"({e}).\nPlease respond with ONLY a valid JSON object "
                        "matching the plan schema (tasks and rationale).\n\n"
                        f"Goal:\n{goal}\n"
                    )
                    response = await self.client.complete(
                        retry_prompt, model=self.client.settings.planner_model
                    )

        raise ValueError(
            f"Failed to parse a valid plan from the LLM after "
            f"{MAX_JSON_RETRIES + 1} attempts ({MAX_JSON_RETRIES} retries): "
            f"{last_error}\nResponse: {response}"
        )
