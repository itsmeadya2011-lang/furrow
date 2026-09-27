from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT, _extract_json
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        logger.info(
            "planner_start",
            goal=goal,
            model=self.client.settings.planner_model,
        )
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = _extract_json(response)
            plan = Plan(**data)
            logger.info(
                "planner_complete",
                model=self.client.settings.planner_model,
                task_count=len(plan.tasks),
            )
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.exception(
                "planner_failure",
                goal=goal,
                error=str(e),
            )
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
