from __future__ import annotations

import json

from furrow.agents.base import BaseAgent, extract_json
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan


class PlannerAgent(BaseAgent):
    """Turns a goal into a validated :class:`~furrow.config.Plan`."""

    model_attr = "planner_model"

    async def plan(self, goal: str) -> Plan:
        response = await self.ask(f"Goal: {goal}", system=PLANNER_PROMPT)
        try:
            return Plan.model_validate(json.loads(extract_json(response)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse plan from LLM: {exc}\nResponse: {response}"
            ) from exc