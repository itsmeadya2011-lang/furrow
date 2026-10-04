from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.json_parse import parse_json_object
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        # Never pass ``settings=None`` through: the client only accepts a real
        # ``Settings`` when one is given explicitly.
        self.client = client or (LLMClient() if settings is None else LLMClient(settings=settings))

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = parse_json_object(response)
            return Plan(**data)
        except ValueError as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e
