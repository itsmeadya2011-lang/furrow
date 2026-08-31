from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = ""
        for attempt in range(3):
            try:
                response = await self.client.complete(
                    prompt, model=self.client.settings.planner_model
                )
                data = json.loads(response)
                plan = Plan(**data)
                logger.info("planning_succeeded", goal=goal, tasks=len(plan.tasks), attempt=attempt)
                return plan
            except json.JSONDecodeError as e:
                logger.warning("planning_failed", goal=goal, attempt=attempt, error=str(e))
                if attempt < 2:
                    prompt = (
                        f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
                        "\nReturn ONLY valid JSON. Do not include markdown or explanations."
                    )
                else:
                    raise ValueError(
                        f"Failed to parse plan from LLM after "
                        f"{attempt + 1} attempts: {e}\nResponse: {response}"
                    )