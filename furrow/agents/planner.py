from __future__ import annotations

import json
from typing import Any

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(
        self,
        goal: str,
        context: str = "",
        task_history: list[TaskModel] | None = None,
        files_changed: list[str] | None = None,
    ) -> Plan:
        """Generate a plan for the given goal with optional context.

        Args:
            goal: The high-level goal to plan for.
            context: Additional context (e.g., test failures from previous cycle).
            task_history: List of tasks from previous cycles with their status and results.
            files_changed: List of files that have been changed in previous cycles.
        """
        prompt_parts = [PLANNER_PROMPT, f"\n\nGoal: {goal}"]

        if context:
            prompt_parts.append(f"\n\nContext:\n{context}")

        # Add task history if available
        if task_history:
            history_lines = ["\n\nPrevious Tasks:"]
            for task in task_history:
                status = task.status
                attempts = task.attempts
                files = ", ".join(task.files_changed) if task.files_changed else "none"
                result = task.result or "no result"
                error = task.error or "none"
                history_lines.append(
                    f"  - {task.id}: {task.description} (status: {status}, attempts: {attempts}, "
                    f"files: {files}, result: {result}, error: {error})"
                )
            prompt_parts.append("\n".join(history_lines))

        # Add files changed in previous cycles
        if files_changed:
            prompt_parts.append(f"\n\nFiles Changed in Previous Cycles:\n{', '.join(files_changed)}")

        prompt = "".join(prompt_parts)

        model = self.settings.planner_model
        logger.debug("planner_request", goal=goal[:200], context_length=len(context), 
                    task_history_count=len(task_history) if task_history else 0,
                    files_changed_count=len(files_changed) if files_changed else 0)
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            plan = parse_plan(response)
            logger.info("planner_response", tasks=len(plan.tasks), rationale=plan.rationale[:200])
            return plan
        except ValueError as first:
            logger.warning("planner_parse_failed_attempt_1", error=str(first))
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
                logger.info("planner_response_retry", tasks=len(plan.tasks), rationale=plan.rationale[:200])
                return plan
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                logger.error("planner_parse_failed_attempt_2", error=str(second), response=raw[:500])
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second
