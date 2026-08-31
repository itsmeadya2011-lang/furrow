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
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info(
                "plan_parsed",
                task_count=len(plan.tasks),
                rationale=plan.rationale,
            )
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                "plan_parse_failed",
                error=str(e),
                response=response,
            )
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
