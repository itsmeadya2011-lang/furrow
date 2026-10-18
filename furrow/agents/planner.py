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

    async def plan(self, goal: str) -> Plan:
        context = await self._build_context()
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nCurrent project structure:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _build_context(self) -> str:
        """Build a concise summary of the workspace for planning context."""
        files = self.client.list_files(self.client.settings.workspace)
        if not files:
            return "(empty workspace)"
        lines = [f"  {f}" for f in files[:50]]
        return "\n".join(lines)