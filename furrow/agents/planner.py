from __future__ import annotations

import time

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        logger.debug("Planner request", model=model, goal_length=len(goal), context_length=len(context))
        start = time.time()
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        elapsed = time.time() - start
        logger.debug("Planner response received", model=model, elapsed_seconds=round(elapsed, 2), response_length=len(response))
        try:
            plan = parse_plan(response)
            logger.info("Plan parsed successfully", tasks=len(plan.tasks), rationale=plan.rationale[:200])
            return plan
        except ValueError as first:
            logger.warning("Planner output parse failed, retrying with stricter prompt", error=str(first))
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
                logger.info("Plan parsed on retry", tasks=len(plan.tasks))
                return plan
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                logger.error("Planner output could not be parsed after retry", attempt1_error=str(first), attempt2_error=str(second), raw_excerpt=raw[:500])
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second
