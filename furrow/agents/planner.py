from __future__ import annotations

import json
import re
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
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = self._parse_json(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Retry once
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            try:
                data = self._parse_json(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e2:
                raise ValueError(f"Failed to parse plan from LLM: {e2}\nResponse: {response}")

    @staticmethod
    def _parse_json(response: str) -> dict:
        text = response.strip()
        # Strip markdown fences
        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        return json.loads(text)
