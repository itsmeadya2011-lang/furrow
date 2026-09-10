from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, history: list[dict[str, Any]] | None = None) -> Plan:
        history_context = ""
        if history:
            history_context = "\nPrevious cycles:\n"
            for entry in history[-3:]:
                history_context += f"- Cycle {entry['cycle']}: {entry['test_result']['summary']}\n"

        prompt = f"{PLANNER_PROMPT.format(history_context=history_context)}\n\nGoal: {goal}"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
