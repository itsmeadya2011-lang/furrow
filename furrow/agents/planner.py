from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.agents.jsonutil import parse_json
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            return Plan(**parse_json(response))
        except (json.JSONDecodeError, ValueError):
            retry_prompt = (
                "Your previous response was not valid JSON. Respond with ONLY a JSON object "
                "having keys 'tasks' and 'rationale'. No markdown fences, no prose.\n\n"
                + prompt
            )
            response = await self.client.complete(
                retry_prompt, model=self.client.settings.planner_model
            )
            try:
                return Plan(**parse_json(response))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse plan from LLM: {e}\nResponse: {response}"
                )