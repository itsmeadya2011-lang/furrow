from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        logger.info("planner.plan", goal=goal)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        cleaned = self._strip_markdown(response)
        try:
            data = json.loads(cleaned)
            logger.info("planner.plan_parsed", task_count=len(data.get("tasks", [])))
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planner.parse_failed", error=str(e), response=cleaned)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {cleaned}") from e

    @staticmethod
    def _strip_markdown(response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return response.strip()
