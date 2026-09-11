from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("furrow.planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> Plan:
        """Parse the LLM response into a Plan, handling markdown-wrapped JSON."""
        # Try direct JSON parse first
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding any JSON-like object in the response
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"Failed to parse plan from LLM response:\n{response[:500]}")