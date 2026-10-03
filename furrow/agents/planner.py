from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from furrow.agents.parsing import coerce_plan, excerpt, parse_json_response
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Turns a goal into an executable :class:`~furrow.config.Plan`."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if settings is not None:
            resolved = settings
        elif client is not None:
            resolved = client.settings
        else:
            resolved = default_settings
        self.client = client or LLMClient(settings=resolved)
        self.settings: Settings = resolved

    async def plan(self, goal: str, context: str = "") -> Plan:
        """Plan ``goal``, optionally using ``context`` from previous cycles.

        The response is parsed leniently (fences, prose, double encoding). One
        corrective retry is attempted before a :class:`ValueError` is raised, so
        a single malformed response cannot kill a cycle.
        """
        prompt = self._build_prompt(goal, context)
        response = await self.client.complete(
            prompt,
            system=PLANNER_SYSTEM_PROMPT,
            model=self.settings.planner_model,
        )
        try:
            return self._to_plan(response)
        except ValueError as exc:
            logger.info("planner response was unusable, retrying once: %s", exc)

        retry_prompt = (
            f"{prompt}\n\nYour previous response could not be read as a plan:\n"
            f"{excerpt(response)}\n\n"
            "Reply again with ONLY the JSON object described above: it must start "
            'with { and end with }, and contain no prose and no markdown fences.'
        )
        retry = await self.client.complete(
            retry_prompt,
            system=PLANNER_SYSTEM_PROMPT,
            model=self.settings.planner_model,
        )
        try:
            return self._to_plan(retry)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse plan from LLM after one retry: {exc}\n"
                f"Response: {excerpt(retry)}"
            ) from exc

    def _build_prompt(self, goal: str, context: str) -> str:
        parts = [
            PLANNER_PROMPT,
            f"Constraints:\n- At most {max(1, self.settings.max_parallel_tasks)} tasks in this plan.",
            f"Goal: {goal}",
        ]
        if context.strip():
            parts.append(
                "Context from previous cycles (what already exists and what failed):\n"
                f"{context.strip()}"
            )
        return "\n\n".join(parts)

    def _to_plan(self, response: str) -> Plan:
        plan = coerce_plan(parse_json_response(response))
        limit = max(1, self.settings.max_parallel_tasks)
        if len(plan.tasks) > limit:
            logger.warning(
                "planner returned %d tasks, keeping the first %d (max_parallel_tasks)",
                len(plan.tasks),
                limit,
            )
            return Plan(tasks=plan.tasks[:limit], rationale=plan.rationale)
        return plan
