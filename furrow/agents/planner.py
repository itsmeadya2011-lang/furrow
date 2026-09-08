from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, previous_failures: str | None = None) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if previous_failures:
            prompt += f"\nPrevious failures to avoid:\n{previous_failures}\n"

        max_attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.complete(prompt, model=self.client.settings.planner_model)
                json_str = self._extract_json(response)
                data = json.loads(json_str)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(
                    "Failed to parse plan from LLM (attempt %d/%d): %s\nResponse: %s",
                    attempt, max_attempts, e, response if 'response' in locals() else '',
                )
                if attempt < max_attempts:
                    await asyncio.sleep(1)

        raise ValueError(
            f"Failed to parse plan from LLM after {max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
