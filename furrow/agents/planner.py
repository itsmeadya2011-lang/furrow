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
        from furrow.config import settings as default_settings

        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str) -> Plan:
        prompt = (
            f"{PLANNER_PROMPT}\n\n"
            f"Max tasks this cycle: {max(1, self.settings.max_parallel_tasks)}\n"
            f"Goal: {goal}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        # Strip common markdown wrappers the LLM sometimes adds.
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            data = json.loads(cleaned)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse plan from LLM: {exc}\nResponse: {response}"
            ) from exc
