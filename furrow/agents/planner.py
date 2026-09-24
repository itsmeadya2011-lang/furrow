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

    async def plan(self, goal: str, project_files: list[str] | None = None, existing_tasks: list | None = None) -> Plan:
        context = ""
        if project_files:
            context += f"\nProject files:\n" + "\n".join(f"  - {f}" for f in project_files[:50])
        if existing_tasks:
            context += "\n\nPrevious tasks:\n"
            for t in existing_tasks[-20:]:
                context += f"  - [{t.status}] {t.id}: {t.description}\n"
                if t.result:
                    context += f"    Result: {t.result[:200]}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")