from __future__ import annotations

import json

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient


def _extract_json(response: str) -> str:
    """Strip markdown code fences and trim whitespace before json.loads."""
    response = response.strip()
    if response.startswith("```"):
        parts = response.split("\n", 1)
        response = parts[1] if len(parts) > 1 else ""
        response = response.removesuffix("```")
    return response.strip()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_response = ""
        for attempt in range(3):
            last_response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = json.loads(_extract_json(last_response))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < 2:
                    prompt = (
                        f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
                        f"\nYour previous response was not valid JSON: {e}\n"
                        "Return valid JSON only.\n"
                    )
                    continue
                raise ValueError(
                    f"Failed to parse plan from LLM: {e}\nResponse: {last_response}"
                )
        raise ValueError(f"Failed to parse plan from LLM\nResponse: {last_response}")
