from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        max_retries = 3
        last_error: Exception | None = None
        response = ""

        for attempt in range(1, max_retries + 1):
            response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = json.loads(response)
                if "tasks" not in data or "rationale" not in data:
                    raise ValueError("Missing required keys 'tasks' or 'rationale'")
                log.info("plan_parsed", attempt=attempt, task_count=len(data["tasks"]))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                log.warning("plan_parse_failed", attempt=attempt, error=str(e))
                if attempt < max_retries:
                    # Ask the LLM to re-emit valid JSON
                    prompt = (
                        f"{PLANNER_PROMPT}\n\n"
                        f"Goal: {goal}\n\n"
                        f"Previous response was not valid JSON. "
                        f"Return ONLY valid JSON matching the specified schema.\n"
                    )

        raise ValueError(
            f"Failed to parse plan from LLM after {max_retries} attempts: {last_error}\n"
            f"Response: {response}"
        )
