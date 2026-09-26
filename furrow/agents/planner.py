from __future__ import annotations

import json
from typing import TYPE_CHECKING

from structlog import get_logger

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    """Translates a high-level goal into a structured Plan with parallelizable tasks."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        logger.debug("planner_response", length=len(response))
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e
