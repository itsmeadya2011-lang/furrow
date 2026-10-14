from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        log.debug("planner.plan.start", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("planner.plan.complete", task_count=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("planner.plan.parse_error", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
