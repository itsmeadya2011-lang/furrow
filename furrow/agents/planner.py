from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger("planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        log.info("planning", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("plan_parsed", num_tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("plan_parse_failed", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
