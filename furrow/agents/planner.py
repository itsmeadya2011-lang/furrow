from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        log.info("planner.request", goal=goal)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("planner.response", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("planner.parse_failed", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
