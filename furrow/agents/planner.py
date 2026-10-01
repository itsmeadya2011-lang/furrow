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

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        max_parallel = self.client.settings.max_parallel_tasks
        prompt = (
            f"{PLANNER_PROMPT}\n\n"
            f"Maximum number of parallel tasks: {max_parallel}\n\n"
            f"Goal: {goal}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.planner_model
        )
        try:
            data = self._parse_json(response)
            plan = Plan(**data)
            # Enforce max_parallel_tasks on the planned tasks.
            if len(plan.tasks) > max_parallel:
                logger.warning(
                    "plan_truncated",
                    requested=len(plan.tasks),
                    max_parallel=max_parallel,
                )
                plan.tasks = plan.tasks[:max_parallel]
            logger.info("plan_created", num_tasks=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse plan from LLM: {e}\nResponse: {response}"
            ) from e

    @staticmethod
    def _parse_json(response: str) -> dict:
        """Extract a JSON dict from the LLM response.

        Strips markdown code fences and handles leading text before the JSON.
        """
        cleaned = response.strip()
        # Strip surrounding code fences.
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback: find the first JSON object in the text.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data

        raise json.JSONDecodeError("No JSON object found in response", response, 0)
