from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, State
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def plan(self, goal: str, state: State | None = None) -> Plan:
        context = ""
        if state and state.cycles:
            context = f"\n\nPrevious cycles: {len(state.cycles)}"
            if state.completed_tasks:
                context += f"\nCompleted tasks: {len(state.completed_tasks)}"
            if state.failed_tasks:
                context += f"\nFailed tasks: {len(state.failed_tasks)}"
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
