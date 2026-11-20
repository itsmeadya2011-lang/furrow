from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger, log_event, log_error


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self._log = get_logger("furrow.planner")

    async def plan(self, goal: str, context: str = "") -> Plan:
        log_event(self._log, "planner.plan_started", goal=goal[:200], has_context=bool(context))
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            plan = parse_plan(response)
            log_event(self._log, "planner.plan_created", tasks=len(plan.tasks), rationale=plan.rationale[:200])
            return plan
        except ValueError as first:
            log_event(self._log, "planner.parse_failed_retry", error=str(first)[:200])
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
                log_event(self._log, "planner.plan_created_retry", tasks=len(plan.tasks), rationale=plan.rationale[:200])
                return plan
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                log_error(self._log, "planner.parse_failed_final", second, response_preview=raw[:500])
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second
