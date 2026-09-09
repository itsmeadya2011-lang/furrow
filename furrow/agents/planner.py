from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        prompt = self._build_prompt(goal, context)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "planner_json_parse_failed",
                attempt=1,
                error=str(e),
                response_preview=response[:500],
            )
            # Retry with a simpler prompt that asks for just the JSON.
            retry_prompt = (
                f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
                + (self._format_context(context) if context else "")
                + "\n\nReturn ONLY valid JSON. No markdown, no explanation."
            )
            retry_response = await self.client.complete(retry_prompt, model=self.client.settings.planner_model)
            try:
                data = json.loads(retry_response)
                logger.info("planner_json_parse_succeeded", attempt=2)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e2:
                logger.error("planner_json_parse_failed_final", error=str(e2))
                raise ValueError(
                    f"Failed to parse plan from LLM after retry: {e2}\nResponse: {retry_response}"
                )

    def _build_prompt(self, goal: str, context: dict[str, Any] | None) -> str:
        base = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if context:
            base += self._format_context(context)
        return base

    def _format_context(self, context: dict[str, Any]) -> str:
        parts: list[str] = []
        failed_tasks = context.get("failed_tasks")
        if failed_tasks:
            parts.append("Previously failed tasks:")
            for t in failed_tasks:
                parts.append(f"  - {t}")
        test_results = context.get("test_results")
        if test_results:
            parts.append("Previous test results:")
            for r in test_results:
                parts.append(f"  - {r}")
        if not parts:
            return ""
        return "\n" + "\n".join(parts) + "\n"