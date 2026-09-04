from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_FENCE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    return _FENCE.sub(r"\1", text.strip()).strip()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(_strip_fences(response))
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            return Plan(
                tasks=[
                    TaskModel(
                        id="1",
                        description=f"Reformat the previous plan as valid JSON matching the schema: {{tasks: [{{id, description, files, dependencies}}]}}. Previous attempt:\n{response[:500]}",
                    )
                ],
                rationale="Recovering from malformed JSON output from the planner LLM.",
            )
