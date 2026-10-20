from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _extract_json(text: str) -> str:
    text = text.strip()
    # strip markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # extract outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or default_settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        strict = (
            "Your previous response was not valid JSON. Respond with ONLY a JSON object "
            'matching this schema: {"tasks":[{"id":str,"description":str,'
            '"files":[str],"dependencies":[str]}],"rationale":str}'
        )
        last_err: Exception | None = None
        last_response = ""
        for attempt in range(2):
            p = prompt if attempt == 0 else prompt + "\n" + strict
            response = await self.client.complete(p, model=self.client.settings.planner_model)
            last_response = response
            try:
                data = json.loads(_extract_json(response))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt == 1:
                    raise ValueError(
                        f"Failed to parse plan from LLM: {e}\nResponse: {response[:500]}"
                    ) from e
        raise ValueError(f"Failed to parse plan: {last_err}\nResponse: {last_response[:500]}")
