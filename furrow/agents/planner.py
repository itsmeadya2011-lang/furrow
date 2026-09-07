from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, history: list[dict[str, Any]] | None = None) -> Plan:
        history_block = ""
        if history:
            history_block = "\n\nPrevious cycles:\n" + json.dumps(history, indent=2)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{history_block}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
