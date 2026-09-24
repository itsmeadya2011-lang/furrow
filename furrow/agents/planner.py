from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\n{context}\n\nGoal: {goal}\n"
        logger.debug("planner_request", goal=goal[:100], context_length=len(context))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        logger.debug("planner_response", response_length=len(response))
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_created", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
