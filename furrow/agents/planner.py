from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.exceptions import PlanParseError
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        logger.info("Planning", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        except Exception as e:
            logger.error("Planning failed", error=str(e))
            raise PlanParseError(f"LLM call failed: {e}") from e

        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("Plan created", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse plan", response=response[:200])
            raise PlanParseError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e