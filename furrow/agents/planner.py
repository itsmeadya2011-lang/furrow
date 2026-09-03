from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        data = _extract_json(response)
        plan = Plan(**data)
        plan = _normalize_plan(plan)
        return plan


def _extract_json(response: str) -> dict:
    """Parse JSON from an LLM response that may wrap it in markdown fences."""
    text = response.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first fence line
        lines = lines[1:]
        # Drop closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse plan JSON from LLM: {e}\nResponse: {response}"
        ) from e
    if not isinstance(result, dict):
        raise ValueError(f"Plan JSON must be an object, got {type(result).__name__}")
    return result


def _normalize_plan(plan: Plan) -> Plan:
    """Validate and normalize a Plan in place.

    - Assigns missing IDs (1, 2, 3, ...).
    - Renames duplicate IDs to "1__dup1", "1__dup2", ....
    - Drops dependencies referencing unknown IDs.
    - Detects cycles among dependencies and drops the back edges.
    """
    seen: set[str] = set()
    for idx, task in enumerate(plan.tasks, start=1):
        if not task.id:
            task.id = str(idx)
        # De-dup IDs while keeping task references intact.
        if task.id in seen:
            base = task.id
            n = 1
            while f"{base}__dup{n}" in seen:
                n += 1
            task.id = f"{base}__dup{n}"
        seen.add(task.id)

    valid_ids = {t.id for t in plan.tasks}

    # Filter invalid dependencies.
    for task in plan.tasks:
        if task.dependencies:
            task.dependencies = [d for d in task.dependencies if d in valid_ids]

    # Drop cyclic dependencies (keep only edges that don't form a cycle).
    for task in plan.tasks:
        if not task.dependencies:
            continue
        kept: list[str] = []
        for dep in task.dependencies:
            # Tentatively add; detect cycle via DFS over kept deps.
            if not _would_cycle(task.id, dep, plan.tasks, kept):
                kept.append(dep)
        task.dependencies = kept

    return plan


def _would_cycle(task_id: str, dep_id: str, tasks: list, kept_for_task: list[str]) -> bool:
    """Return True if adding dep_id as a dependency of task_id would create a cycle.

    A cycle exists if task_id is reachable from dep_id through the tentative
    dependency graph (including the kept_for_task entries we already added).
    """
    adj: dict[str, list[str]] = {t.id: list(t.dependencies or []) for t in tasks}
    adj[task_id] = list(kept_for_task) + [dep_id]
    # BFS from dep_id; if we reach task_id, that's a cycle.
    visited: set[str] = set()
    stack = [dep_id]
    while stack:
        node = stack.pop()
        if node == task_id:
            return True
        if node in visited:
            continue
        visited.add(node)
        for nxt in adj.get(node, []):
            if nxt not in visited:
                stack.append(nxt)
    return False