from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_TASKS = 5
MAX_PLAN_ATTEMPTS = 2

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)
_RETRY_HINT = (
    "\n\nYour previous reply could not be parsed. Reply again with BARE JSON only: no markdown code"
    ' fences, no prose before or after. Use exactly one JSON object with a "tasks" array (objects with'
    ' "id", "description", "files", "dependencies") and a "rationale" string.'
)


def _iter_object_blocks(text: str) -> list[str]:
    """Return every balanced ``{...}`` region of ``text``, outermost candidates first."""
    blocks: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start : index + 1])
                    break
    return blocks


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of ``text``, tolerating code fences and surrounding prose.

    Raises ``ValueError`` when no JSON object can be recovered. The caller is responsible for
    including the raw response in any error message it surfaces.
    """
    stripped = text.strip()
    candidates: list[str] = [stripped] if stripped else []
    candidates.extend(block.strip() for block in _FENCE_RE.findall(stripped))
    candidates.extend(_iter_object_blocks(stripped))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("no JSON object could be recovered from the response")


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_task(raw: Any, index: int) -> TaskModel | None:
    """Turn one raw plan entry into a ``TaskModel``, or ``None`` when it is unusable."""
    position = index + 1
    if not isinstance(raw, dict):
        logger.warning("Dropping plan entry #%d: expected an object, got %s", position, type(raw).__name__)
        return None
    description = raw.get("description") or raw.get("task") or raw.get("title") or ""
    text = str(description).strip()
    if not text:
        logger.warning("Dropping plan entry #%d: description is empty", position)
        return None
    status = str(raw.get("status") or "pending").strip() or "pending"
    return TaskModel(
        id=str(raw.get("id") or "").strip(),
        description=text,
        files=_as_str_list(raw.get("files")),
        dependencies=_as_str_list(raw.get("dependencies")),
        status=status,
    )


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        model = self.client.settings.planner_model
        response = ""
        failure = "the model returned no response"
        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            request = prompt if attempt == 1 else prompt + _RETRY_HINT
            response = await self.client.complete(request, model=model)
            try:
                return self._build_plan(extract_json_object(response))
            except ValueError as exc:
                failure = str(exc)
                logger.warning(
                    "Planner produced an unusable plan on attempt %d/%d: %s",
                    attempt,
                    MAX_PLAN_ATTEMPTS,
                    failure,
                )
        raise ValueError(f"Failed to parse plan from LLM: {failure}\nResponse: {response}")

    def _build_plan(self, data: dict[str, Any]) -> Plan:
        """Validate and sanitise a decoded plan payload."""
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("the plan JSON has no 'tasks' array")
        tasks = self._coerce_tasks(raw_tasks, self._max_tasks())
        self._prune_dependencies(tasks)
        rationale = str(data.get("rationale") or "").strip()
        if not tasks:
            logger.warning("Plan for the goal ended up with zero usable tasks")
            rationale = rationale or "The planner returned no actionable tasks for this goal."
        return Plan(tasks=tasks, rationale=rationale)

    def _coerce_tasks(self, raw_tasks: list[Any], limit: int) -> list[TaskModel]:
        tasks: list[TaskModel] = []
        dropped = 0
        for index, raw in enumerate(raw_tasks):
            task = _coerce_task(raw, index)
            if task is None:
                dropped += 1
                continue
            tasks.append(task)
        if len(tasks) > limit:
            dropped += len(tasks) - limit
            logger.warning(
                "Plan returned %d usable tasks; capping at max_parallel_tasks=%d",
                len(tasks),
                limit,
            )
            tasks = tasks[:limit]
        if dropped:
            logger.warning("Plan dropped %d unusable task(s), kept %d", dropped, len(tasks))
        self._assign_ids(tasks)
        return tasks

    def _assign_ids(self, tasks: list[TaskModel]) -> None:
        used: set[str] = set()
        for position, task in enumerate(tasks, start=1):
            base = task.id or f"task-{position}"
            candidate = base
            suffix = 2
            while candidate in used:
                candidate = f"{base}-{suffix}"
                suffix += 1
            if candidate != task.id:
                if task.id:
                    logger.warning("Duplicated plan id %r, renamed to %r", task.id, candidate)
                else:
                    logger.warning("Plan entry #%d had no id, assigned %r", position, candidate)
            used.add(candidate)
            task.id = candidate

    def _prune_dependencies(self, tasks: list[TaskModel]) -> None:
        known = {task.id for task in tasks}
        for task in tasks:
            kept: list[str] = []
            for dependency in task.dependencies:
                if dependency == task.id:
                    logger.warning("Task %r depends on itself; dropping the dependency", task.id)
                elif dependency not in known:
                    logger.warning(
                        "Task %r depends on unknown task %r; dropping the dependency",
                        task.id,
                        dependency,
                    )
                else:
                    kept.append(dependency)
            task.dependencies = kept

    def _max_tasks(self) -> int:
        configured = getattr(self.client.settings, "max_parallel_tasks", DEFAULT_MAX_TASKS)
        try:
            limit = int(configured)
        except (TypeError, ValueError):
            logger.warning(
                "max_parallel_tasks=%r is not an integer; falling back to %d",
                configured,
                DEFAULT_MAX_TASKS,
            )
            return DEFAULT_MAX_TASKS
        if limit < 1:
            logger.warning("max_parallel_tasks=%d is not positive; using %d", limit, DEFAULT_MAX_TASKS)
            return DEFAULT_MAX_TASKS
        return limit