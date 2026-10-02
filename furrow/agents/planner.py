from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    for fence in ("```json", "```"):
        cleaned = cleaned.removeprefix(fence)
    for fence in ("```json", "```"):
        cleaned = cleaned.removesuffix(fence)
    return cleaned.strip()


class PlannerAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        cleaned = _strip_markdown_fences(response)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.debug("plan.parse_retry", goal=goal)
            try:
                data = json.loads(_strip_markdown_fences(response))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse plan from LLM: {e}\nResponse: {response}"
                ) from e
        plan = Plan(**data)
        logger.info("plan", tasks=len(plan.tasks))
        return plan
