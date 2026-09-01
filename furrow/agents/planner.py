from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        user_prompt = f"Goal: {goal}\n"
        response = await self.client.complete(
            user_prompt,
            system=PLANNER_PROMPT,
            model=self.client.settings.planner_model,
        )
        data = extract_json(response)
        if not isinstance(data, dict):
            raise ValueError(
                f"Failed to parse plan from LLM (no JSON object found).\nResponse: {response}"
            )
        try:
            return Plan(**data)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")