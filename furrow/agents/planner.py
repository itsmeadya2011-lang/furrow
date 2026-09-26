from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    logger = None


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            result = Plan(**data)
            if logger:
                logger.info("plan_created", task_count=len(result.tasks))
            return result
        except (json.JSONDecodeError, ValueError) as e:
            if logger:
                logger.error("plan_parse_error", error=str(e), response_preview=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
