from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_PATTERN.sub("", text, count=2)
    return text.strip()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        cleaned = _strip_code_fence(response)
        try:
            data = json.loads(cleaned)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[PlannerAgent] Failed to parse plan JSON: {e}\nResponse: {response}")
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
