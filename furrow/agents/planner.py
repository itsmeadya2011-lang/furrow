from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_RETRY_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _extract_json(text: str) -> str | None:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        plan = self._try_parse_plan(response)
        if plan is not None:
            return plan

        retry_prompt = f"{PLANNER_RETRY_PROMPT}\n\nGoal: {goal}\n"
        retry_response = await self.client.complete(retry_prompt, model=self.client.settings.planner_model)
        plan = self._try_parse_plan(retry_response)
        if plan is not None:
            return plan

        snippet = response[:500] + ("..." if len(response) > 500 else "")
        raise ValueError(f"Failed to parse plan from LLM after retry. Response snippet: {snippet}")

    def _try_parse_plan(self, response: str) -> Plan | None:
        json_text = _extract_json(response)
        if json_text is None:
            return None
        try:
            data = json.loads(json_text)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            return None