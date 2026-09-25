from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents._util import parse_json_response
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
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        data = parse_json_response(response)
        if not data:
            log.warning("planner_parse_failed", response=response[:200])
            return Plan(tasks=[], rationale="Failed to parse plan from LLM response.")
        try:
            return Plan(**data)
        except (ValueError, TypeError) as e:
            log.warning("planner_schema_invalid", error=str(e))
            return Plan(tasks=[], rationale=f"Invalid plan schema: {e}")
