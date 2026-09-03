from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_JSON_RETRIES: int = 1

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _extract_json(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1)
    return text


async def _parse_json_response(
    response: str,
    client: LLMClient,
    original_prompt: str,
    model: str | None,
) -> dict[str, Any]:
    try:
        return json.loads(_extract_json(response))
    except (json.JSONDecodeError, ValueError):
        pass

    corrective = (
        "Your previous response was not valid JSON. "
        "Return ONLY valid JSON, no markdown.\n\n"
        f"Prior response:\n{response}"
    )
    retried = await client.complete(corrective, model=model)
    try:
        return json.loads(_extract_json(retried))
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        data = await _parse_json_response(
            response, self.client, prompt, self.client.settings.planner_model
        )
        return Plan(**data)