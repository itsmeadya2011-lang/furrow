from __future__ import annotations

import logging
from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Goal cannot be empty")
        
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        logger.debug("Planning with model %s, goal length: %d", model, len(goal))
        
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            plan = parse_plan(response)
            self._validate_plan(plan)
            logger.info("Plan created: %d tasks, rationale: %s", len(plan.tasks), plan.rationale[:100])
            return plan
        except ValueError as first:
            logger.warning("First planner attempt failed: %s", first)
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
                self._validate_plan(plan)
                logger.info("Plan created on retry: %d tasks", len(plan.tasks))
                return plan
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                logger.error("Planner failed after 2 attempts. First: %s, Second: %s, Response: %s", 
                           first, second, raw[:500])
                raise ValueError(
                    "Planner output could not be parsed after 2 attempts. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    def _validate_plan(self, plan: Plan) -> None:
        """Validate that the plan has reasonable structure."""
        if not plan.tasks:
            logger.warning("Plan has no tasks")
            return
        
        for i, task in enumerate(plan.tasks):
            if not task.description or not task.description.strip():
                raise ValueError(f"Task {i} has empty description")
            if not task.id:
                raise ValueError(f"Task {i} has empty id")
            # Check for duplicate task IDs
            ids = [t.id for t in plan.tasks]
            if len(ids) != len(set(ids)):
                raise ValueError("Plan contains duplicate task IDs")
        
        # Validate dependencies reference existing tasks
        task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep}")
