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
        logger.info("planning_started", goal=goal)

        max_parallel = self.client.settings.max_parallel_tasks
        prompt = (
            f"{PLANNER_PROMPT.replace('{max_parallel_tasks}', str(max_parallel))}\n\nGoal: {goal}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)

        plan = self._parse_plan(response)
        if plan is None:
            logger.warning("plan_parse_failed", response_preview=response[:200])
            logger.info("retrying_plan_with_clarification")
            clarify_prompt = (
                f"{PLANNER_PROMPT.replace('{max_parallel_tasks}', str(max_parallel))}\n\n"
                f"Goal: {goal}\n\n"
                f"The previous response was not valid JSON. Please return valid JSON only, "
                f"with no markdown formatting or explanations.\n"
                f"Previous response: {response}"
            )
            response = await self.client.complete(
                clarify_prompt, model=self.client.settings.planner_model
            )
            plan = self._parse_plan(response)
            if plan is None:
                logger.error("plan_parse_failed_after_retry", response_preview=response[:200])
                raise ValueError(
                    f"Failed to parse plan from LLM: invalid JSON\nResponse: {response}"
                )

        logger.info("plan_created", num_tasks=len(plan.tasks))
        return plan

    def _parse_plan(self, response: str) -> Plan | None:
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            return None
