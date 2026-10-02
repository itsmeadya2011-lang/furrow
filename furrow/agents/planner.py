from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _strip_markdown_fence(text: str) -> str:
    """Strip markdown code fences from LLM text output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
    elif text.startswith("```"):
        text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(_strip_markdown_fence(response))
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            # Try to salvage by extracting JSON with regex
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                pass
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
