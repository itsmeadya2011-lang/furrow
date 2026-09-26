from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings or default_settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        logger.info("planning_started", goal=goal)
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planning_completed", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("plan_parse_failed", error=str(e))
            raise ValueError(f"Failed to parse plan JSON: {e}") from e
