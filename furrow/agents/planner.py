from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self._settings = settings or default_settings

    async def plan(self, goal: str) -> Plan:
        logger.info("planner.start", goal=goal[:100])
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        planner_model = getattr(self.client, 'settings', self._settings).planner_model
        response = await self.client.complete(prompt, model=planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planner.completed", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planner.parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
