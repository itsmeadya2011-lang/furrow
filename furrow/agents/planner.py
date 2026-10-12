from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_PARSE_RETRY_INSTRUCTION = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY valid JSON with no markdown code fences and no explanation.\n"
)


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        cleaned = cleaned[newline + 1 :] if newline != -1 else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return cleaned


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self._attempts = 0

    async def plan(self, goal: str) -> Plan:
        self._attempts = 0
        return await self._plan_once(goal)

    @retry(
        retry=retry_if_exception_type(ValueError),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _plan_once(self, goal: str) -> Plan:
        self._attempts += 1
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if self._attempts > 1:
            prompt += _PARSE_RETRY_INSTRUCTION
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(_extract_json(response))
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
