from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _fallback_plan(retry_state: RetryCallState) -> Plan:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome else None
    msg = str(exc) if exc else "Unknown parsing failure"
    return Plan(
        tasks=[TaskModel(id="1", description=msg)],
        rationale="Minimal fallback plan generated after LLM parsing failures.",
    )


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
        retry_error_callback=_fallback_plan,
    )
    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return Plan(**data)
                except (json.JSONDecodeError, ValueError):
                    pass
            raise ValueError(f"Failed to parse plan from LLM\nResponse: {response}")
