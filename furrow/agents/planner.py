from __future__ import annotations

from typing import Any

from furrow.agents.base import BaseAgent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import extract_json


class PlannerAgent(BaseAgent):
    model_key = "planner_model"

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.model)
        return _parse_plan(response)


def _parse_plan(response: str) -> Plan:
    """Build a Plan from an LLM response: fenced JSON, a bare list, or one task dict."""
    data = extract_json(response)
    if isinstance(data, list):
        data = {"tasks": data}
    elif isinstance(data, dict) and "tasks" not in data and "description" in data:
        data = {"tasks": [data]}
    if not isinstance(data, dict):
        raise ValueError(f"Plan must be an object or a list of tasks, got {type(data).__name__}")

    raw_tasks = data.get("tasks") or []
    if not isinstance(raw_tasks, list):
        raise ValueError(f"Plan 'tasks' must be a list, got {type(raw_tasks).__name__}")
    tasks = [_build_task(raw, i) for i, raw in enumerate(raw_tasks)]
    _validate(tasks)
    return Plan(tasks=tasks, rationale=str(data.get("rationale") or ""))


def _build_task(raw: Any, index: int) -> TaskModel:
    if isinstance(raw, str):
        raw = {"description": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Task at index {index} must be an object, got {type(raw).__name__}")
    description = str(raw.get("description") or "").strip()
    if not description:
        raise ValueError(f"Task at index {index} has no description")
    return TaskModel(
        id=str(raw.get("id") if raw.get("id") not in (None, "") else index + 1),
        description=description,
        files=[str(f) for f in raw.get("files") or []],
        dependencies=[str(d) for d in raw.get("dependencies") or []],
    )


def _validate(tasks: list[TaskModel]) -> None:
    """Reject ids the scheduler cannot key on, and drop deps on unknown tasks."""
    ids = [t.id for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"Plan has duplicate task ids: {', '.join(dupes)}")
    known = set(ids)
    for task in tasks:
        if task.id in task.dependencies:
            raise ValueError(f"Task {task.id} lists itself in dependencies")
        task.dependencies = [d for d in task.dependencies if d in known]
