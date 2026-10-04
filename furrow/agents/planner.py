from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger()


def extract_json(text: str) -> str:
    """Pull the first JSON object out of a possibly noisy LLM response.

    Handles markdown code fences (``` or ```json) and prose wrapped around the
    object. Raises ValueError when no object can be located.
    """
    stripped = text.strip()

    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 2:
            block = parts[1].strip()
            # Drop a leading language tag such as "json" or "JSON".
            head, sep, rest = block.partition("\n")
            if sep and "{" not in head:
                block = rest.strip()
            start, end = block.find("{"), block.rfind("}")
            if start != -1 and end > start:
                return block[start : end + 1]

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return stripped[start : end + 1]


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        log.debug("plan.request", goal=goal, model=self.client.settings.planner_model)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(extract_json(response))
            plan = Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("plan.parse_failed", error=str(e))
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
        log.debug("plan.parsed", tasks=len(plan.tasks), rationale=plan.rationale)
        return plan