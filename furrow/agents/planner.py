from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()

    async def plan(self, goal: str) -> Plan:
        logger.info("planner_start", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        logger.debug("planner_raw_response", response=response[:200])
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planner_done", tasks_count=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
