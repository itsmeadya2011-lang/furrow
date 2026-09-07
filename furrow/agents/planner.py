from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or (LLMClient(settings=settings) if settings is not None else LLMClient())

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        log.debug("planner.plan.start", goal=goal)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("planner.plan.ok", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("planner.plan.parse_failed", error=str(e))
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
