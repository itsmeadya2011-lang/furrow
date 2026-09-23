from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        logger.info("planner.planning", goal_length=len(goal), context_length=len(context))
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nContext:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planner.planned", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planner.parse_failed", error=str(e))
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")