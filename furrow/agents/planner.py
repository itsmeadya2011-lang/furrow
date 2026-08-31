from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        logger.info("planner.start", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        except Exception as e:
            logger.error("planner.failure", error=str(e))
            raise
        try:
            data = json.loads(response)
            plan = Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planner.parse_error", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
        logger.info("planner.success", tasks=len(plan.tasks))
        return plan