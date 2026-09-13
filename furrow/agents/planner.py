from __future__ import annotations

import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

logger = structlog.get_logger()

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        logger.info("planning_start", goal=goal[:100])
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_complete", task_count=len(plan.tasks), rationale=plan.rationale[:100])
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
