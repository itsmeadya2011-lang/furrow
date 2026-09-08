from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, original_goal: str | None = None) -> Plan:
        file_tree = "\n".join(self.client.list_files(self.client.settings.workspace))
        context = f"Goal: {goal}\n"
        if original_goal and original_goal != goal:
            context += f"Original goal: {original_goal}\n"
        prompt = f"{PLANNER_PROMPT}\n\nWorkspace file tree:\n{file_tree}\n\n{context}"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
