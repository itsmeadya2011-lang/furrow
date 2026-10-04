from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.jsonparse import extract_json
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_REPAIR_NUDGE = "\n\nYour previous response was not valid JSON. Return JSON only, no markdown, no prose."


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if context:
            prompt += f"\nWork already done:\n{context}\n"

        last_error: Exception | None = None
        response = ""
        for _attempt in range(2):
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            try:
                return Plan(**json.loads(extract_json(response)))
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                prompt += _REPAIR_NUDGE

        raise ValueError(f"Failed to parse plan from LLM: {last_error}\nResponse: {response[:1000]}")