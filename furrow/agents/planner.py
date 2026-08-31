from __future__ import annotations

import json

from furrow.agents._json import _extract_json
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        files = self.client.list_files(self.client.settings.workspace)
        repo_context = "\n".join(f"  - {f}" for f in files[:200]) if files else "  (empty or unreadable)"
        prompt = f"{PLANNER_PROMPT}\n\nExisting repo files:\n{repo_context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(_extract_json(response))
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
