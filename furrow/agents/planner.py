from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_retries = 3

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"

        for attempt in range(self.max_retries):
            try:
                response = await self.client.complete(prompt, model=self.client.settings.planner_model)
                data = json.loads(response)
                plan = Plan(**data)
                logger.info("Plan created", tasks=len(plan.tasks), rationale=plan.rationale)
                return plan
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse plan, retrying", attempt=attempt + 1, error=str(e))
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to parse plan from LLM after {self.max_retries} attempts: {e}\nResponse: {response}")
                # Add feedback to prompt for retry
                prompt += f"\n\nPrevious response was invalid JSON. Error: {e}\nPlease return valid JSON only."
        raise ValueError("Max retries exceeded")