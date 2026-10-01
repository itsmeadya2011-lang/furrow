from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        log.debug("planner.planning", goal=goal)
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        log.debug("planner.llm_response", response_length=len(response))
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            log.error("planner.parse_failed", error=str(e))
            raise ValueError(
                f"Failed to parse plan from LLM: {e}\nResponse: {response}"
            )
