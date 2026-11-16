from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self._skip_ids: set[str] = set()

    @property
    def skip_ids(self) -> set[str]:
        """Task IDs that should be skipped (already completed identically)."""
        return self._skip_ids

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            return parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    def merge_plan(self, tasks: list[TaskModel], plan: Plan) -> None:
        """Merge a new plan into the existing task list.

        Preserves terminal states; only retry-eligible failed tasks may be
        reset to pending. Blocked tasks remain blocked so that retry budgets
        and dependency outcomes are not lost across cycles.
        """
        self._skip_ids = set()
        by_id = {t.id: t for t in tasks}
        max_retries = max(0, self.settings.max_task_retries)

        for index, pt in enumerate(plan.tasks):
            old = by_id.get(pt.id)
            if old is not None:
                old_desc = old.description
                old_files = list(old.files)
                # Preserve terminal states; only retry-eligible failed tasks may
                # be reset to pending. Blocked tasks must remain blocked so that
                # retry budgets and dependency outcomes are not lost across cycles.
                if old.status == "completed":
                    if old_desc == pt.description and old_files == pt.files:
                        pt.status = "completed"
                        self._skip_ids.add(pt.id)
                    else:
                        pt.status = "pending"
                elif old.status == "failed":
                    if old.attempts <= max_retries:
                        pt.status = "pending"
                    else:
                        pt.status = "failed"
                elif old.status == "blocked":
                    pt.status = "blocked"
                else:
                    pt.status = "pending"
                old.description = pt.description
                old.files = pt.files
                old.dependencies = pt.dependencies
                old.status = pt.status
                # A freshly planned task carries result=None and error=None,
                # so never let that wipe a result or error already recorded by
                # a previous cycle. Only adopt non-None values from the plan.
                if pt.result is not None:
                    old.result = pt.result
                if pt.error is not None:
                    old.error = pt.error
                plan.tasks[index] = old
            else:
                tasks.append(pt)

    def plan_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into waves based on dependencies.

        Tasks in the same wave can run in parallel. A task is in a wave when
        all its dependencies are in earlier waves (or not in the current plan).
        """
        ids = {t.id for t in tasks}
        waves: list[list[TaskModel]] = []
        done: set[str] = set()
        remaining = list(tasks)
        while remaining:
            wave = [t for t in remaining if all(d in done or d not in ids for d in t.dependencies)]
            if not wave:
                for t in remaining:
                    t.status = "blocked"
                break
            waves.append(wave)
            for t in wave:
                done.add(t.id)
            remaining = [t for t in remaining if t not in wave]
        return waves
