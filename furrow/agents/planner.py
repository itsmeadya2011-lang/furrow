from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        if not goal or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            plan = parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

        self._validate_plan(plan)
        return plan

    @staticmethod
    def _validate_plan(plan: Plan) -> None:
        """Reject plans with duplicate IDs or dependencies that reference unknown tasks."""
        ids = {t.id for t in plan.tasks}
        if len(ids) != len(plan.tasks):
            seen: set[str] = set()
            dupes: list[str] = []
            for t in plan.tasks:
                if t.id in seen:
                    dupes.append(t.id)
                seen.add(t.id)
            raise ValueError(f"plan contains duplicate task IDs: {sorted(set(dupes))}")
        for t in plan.tasks:
            for dep in t.dependencies:
                if dep not in ids:
                    raise ValueError(
                        f"task {t.id} depends on unknown task {dep!r}"
                    )
