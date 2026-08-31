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

    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        context_str = ""
        if context:
            context_str = f"\nContext:\n- Original goal: {context.get('original_goal', goal)}\n- Cycle: {context.get('cycle', 1)}\n"
            if context.get("history"):
                context_str += "- Recent history:\n"
                for entry in context["history"]:
                    status = "passed" if entry.get("passed") else "failed"
                    context_str += f"  - Cycle {entry['cycle']}: {status}\n"
        prompt = f"{PLANNER_PROMPT.format(context=context_str)}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
