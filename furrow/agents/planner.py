from __future__ import annotations

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
        logger.info("planner_started", model=model, goal_length=len(goal))
        try:
            response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
            plan = parse_plan(response)
            logger.info("planner_finished", tasks=len(plan.tasks), rationale=plan.rationale[:200])
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
                logger.info("planner_finished_retry", tasks=len(plan.tasks))
                return plan
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                logger.error("planner_parse_failed_attempt_2", error=str(second), response=raw[:500])
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second