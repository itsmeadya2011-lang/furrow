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
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"

        for attempt in range(1, self.client.settings.max_retries + 1):
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)

            plan = self._parse_plan(response)
            if plan is not None:
                return plan

            # On parse failure, add an explicit instruction and retry
            prompt = (
                f"{PLANNER_PROMPT}\n\n"
                f"IMPORTANT: Your previous response was not valid JSON. "
                f"Respond with ONLY valid JSON matching the required schema. "
                f"Goal: {goal}\n"
            )

        raise ValueError(
            f"Failed to parse plan from LLM after {self.client.settings.max_retries} attempts.\n"
            f"Last response: {response}"
        )

    @staticmethod
    def _parse_plan(response: str) -> Plan | None:
        """Attempt to parse an LLM response into a Plan. Returns None on failure."""
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code block
        cleaned = PlannerAgent._extract_json_block(response)
        if cleaned:
            try:
                return Plan(**json.loads(cleaned))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """Extract JSON from a markdown code block if present."""
        import re

        match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
