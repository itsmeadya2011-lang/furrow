from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    MAX_PLAN_ATTEMPTS = 3

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_response = None
        last_error: Exception | None = None

        for attempt in range(self.MAX_PLAN_ATTEMPTS):
            try:
                response = await self.client.complete(
                    prompt, model=self.client.settings.planner_model
                )
                last_response = response
                data = json.loads(response)
                return self._validate_plan(data)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == self.MAX_PLAN_ATTEMPTS - 1:
                    response_text = (
                        last_response
                        if isinstance(last_response, str)
                        else repr(last_response)
                    )
                    raise ValueError(
                        "Failed to obtain a valid plan after "
                        f"{self.MAX_PLAN_ATTEMPTS} attempts "
                        f"(last error: {exc}). Last response: {response_text}"
                    ) from exc

        raise AssertionError("plan retry loop exited unexpectedly")

    @classmethod
    def _validate_plan(cls, data: object) -> Plan:
        if not isinstance(data, dict):
            raise ValueError("Plan JSON must be an object")

        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("Plan tasks must be a list")
        if not tasks:
            return Plan(**data)
        if len(tasks) > 5:
            raise ValueError("Plan must contain between 1 and 5 tasks")

        task_ids: list[str] = []
        for index, task_data in enumerate(tasks):
            if not isinstance(task_data, dict):
                raise ValueError(f"Task {index} must be an object")

            task_id = task_data.get("id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"Task {index} must have a nonempty string id")
            task_ids.append(task_id)

            description = task_data.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Task {index} must have a nonempty string description"
                )

            dependencies = task_data.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError(
                    f"Task {index} dependencies must be a list"
                )
            for dependency_index, dependency in enumerate(dependencies):
                if not isinstance(dependency, str) or not dependency.strip():
                    raise ValueError(
                        f"Task {index} dependency {dependency_index} must be a "
                        "nonempty string id"
                    )

        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Plan task ids must be unique")

        plan = Plan(**data)
        known_ids = set(task_ids)
        tasks_by_id = {task.id: task for task in plan.tasks}

        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency == task.id:
                    raise ValueError(
                        f"Task {task.id!r} cannot depend on itself"
                    )
                if dependency not in known_ids:
                    raise ValueError(
                        f"Task {task.id!r} has unknown dependency {dependency!r}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()
        path: list[str] = []

        def visit(task_id: str) -> None:
            if task_id in visiting:
                cycle_start = path.index(task_id)
                cycle = path[cycle_start:] + [task_id]
                raise ValueError(
                    "Dependency cycle detected: " + " -> ".join(cycle)
                )
            if task_id in visited:
                return

            visiting.add(task_id)
            path.append(task_id)
            for dependency in tasks_by_id[task_id].dependencies:
                visit(dependency)
            path.pop()
            visiting.remove(task_id)
            visited.add(task_id)

        for task in plan.tasks:
            visit(task.id)

        return plan
