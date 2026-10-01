from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = PLANNER_PROMPT.format(goal=goal)
        logger.info("planner_plan", goal=goal[:80])
        response = await self.client.complete(
            prompt, system=PLANNER_SYSTEM, model=self.client.settings.planner_model
        )
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e
