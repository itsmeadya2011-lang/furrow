from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

_MAX_PARSE_RETRIES = 2


def extract_json(text: str) -> dict:
    log.debug("extract_json.attempt", strategy="direct")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    log.debug("extract_json.attempt", strategy="code_block")
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    log.debug("extract_json.attempt", strategy="brace_match")
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    log.error("extract_json.failed", text=text[:500])
    raise ValueError(f"Could not extract valid JSON from text: {text[:500]}")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)

        for attempt in range(_MAX_PARSE_RETRIES + 1):
            try:
                data = extract_json(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                log.warning(
                    "plan.parse_failed",
                    attempt=attempt,
                    error=str(e),
                    response_preview=response[:500],
                )
                if attempt >= _MAX_PARSE_RETRIES:
                    raise ValueError(
                        f"Failed to parse plan from LLM after {_MAX_PARSE_RETRIES + 1} attempts: {e}\nResponse: {response}"
                    )
                retry_prompt = (
                    f"{prompt}\n\nYour previous response was not valid JSON. "
                    f"Error: {e}\nPlease respond with ONLY valid JSON and nothing else."
                )
                response = await self.client.complete(
                    retry_prompt, model=self.client.settings.planner_model
                )

        raise ValueError("Unreachable: planner retry loop exited without return or raise")
