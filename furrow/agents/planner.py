from __future__ import annotations

import json
from typing import Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient, LLMError

SNIPPET_LIMIT = 200
"""Maximum number of characters of the model reply echoed in a PlannerError."""

REPAIR_ATTEMPTS = 1
"""How many times a non-JSON planner reply is re-requested before giving up."""


class PlannerError(LLMError):
    """The planner did not return a usable task list."""


def _snippet(value: object) -> str:
    """Render ``value`` as a short, log-safe string for error messages."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    text = text.strip()
    if len(text) <= SNIPPET_LIMIT:
        return text
    return text[:SNIPPET_LIMIT] + "..."


def _reaches(graph: dict[str, list[str]], start: str, target: str) -> bool:
    """Whether ``target`` is reachable from ``start`` by following ``graph``."""
    pending = [start]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, []))
    return False


class PlannerAgent:
    """Turns a goal into a normalized :class:`~furrow.config.Plan`."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings = settings if settings is not None else self.client.settings

    async def plan(self, goal: str, context: str = "") -> Plan:
        """Ask the model for a plan and normalize it.

        Raises:
            PlannerError: if the reply is not JSON, has no ``tasks`` list, or the
                normalized task list is empty. The message carries a truncated
                snippet of the offending reply.
        """
        prompt = self._build_prompt(goal, context)
        data: Any = await self.client.complete_json(
            prompt,
            model=self.settings.planner_model,
            repair_attempts=REPAIR_ATTEMPTS,
        )
        if not isinstance(data, dict):
            raise PlannerError(
                f"Planner reply was {type(data).__name__}, expected a JSON object. "
                f"Raw response: {_snippet(data)}"
            )
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise PlannerError(
                f"Planner returned no usable 'tasks' list. Raw response: {_snippet(data)}"
            )
        try:
            plan = Plan.model_validate(data)
        except ValueError as error:
            raise PlannerError(
                f"Planner reply did not match the plan schema: {error}. "
                f"Raw response: {_snippet(data)}"
            ) from error
        normalized = self.normalize(plan)
        if not normalized.tasks:
            raise PlannerError(
                f"Planner returned no usable 'tasks' list. Raw response: {_snippet(data)}"
            )
        return normalized

    def normalize(self, plan: Plan) -> Plan:
        """Return a fresh, safe plan: bounded size, sequential ids, acyclic deps."""
        tasks = list(plan.tasks)[: self._task_limit(len(plan.tasks))]
        ids = self._task_ids(tasks)
        known_ids = set(ids)
        dependencies: dict[str, list[str]] = {}
        copies: list[TaskModel] = []
        for task_id, task in zip(ids, tasks, strict=True):
            pruned: list[str] = []
            for dep in self._clean_strings(task.dependencies):
                if dep not in known_ids or dep == task_id or dep in pruned:
                    continue
                # A dependency that (transitively) needs this task is a cycle: drop it.
                if _reaches({**dependencies, task_id: pruned}, dep, task_id):
                    continue
                pruned.append(dep)
            dependencies[task_id] = pruned
            copies.append(
                task.model_copy(
                    update={
                        "id": task_id,
                        "files": self._clean_strings(task.files),
                        "dependencies": list(pruned),
                    },
                    deep=True,
                )
            )
        return Plan(tasks=copies, rationale=plan.rationale or "")

    def _build_prompt(self, goal: str, context: str = "") -> str:
        sections = [PLANNER_PROMPT, f"\nGoal: {goal.strip()}"]
        if context.strip():
            sections.append(f"\nProject context:\n{context.strip()}")
        return "".join(sections)

    def _task_limit(self, count: int) -> int:
        parallel = max(1, int(self.settings.max_parallel_tasks))
        return max(1, min(count, parallel))

    @staticmethod
    def _clean_strings(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        return [value for value in values if isinstance(value, str) and value]

    @staticmethod
    def _task_ids(tasks: list[TaskModel]) -> list[str]:
        """Keep existing ids only when they are all present and unique."""
        raw_ids = [task.id for task in tasks]
        if all(raw_ids) and len(set(raw_ids)) == len(raw_ids):
            return [task_id for task_id in raw_ids]
        return [str(index) for index in range(1, len(tasks) + 1)]


__all__ = ["PlannerAgent", "PlannerError"]