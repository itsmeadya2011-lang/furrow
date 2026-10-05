from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from furrow.agents._shared import as_str_list, as_text, truncate
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

#: Characters of the raw model response echoed back in a parse-failure message.
MAX_ERROR_PREVIEW_CHARS = 1000
#: Upper bound on a description synthesised for a task that carries no prose field.
MAX_SYNTHETIC_DESCRIPTION_CHARS = 300
#: Keys a model plausibly uses for the task text, in preference order.
DESCRIPTION_KEYS = ("description", "task", "summary", "title", "name", "details")
#: Keys a model plausibly uses for the dependency list.
DEPENDENCY_KEYS = ("dependencies", "depends_on", "deps")
#: Stand-in rationale so ``Plan`` validates even when the model omits the field.
DEFAULT_RATIONALE = "(the planner returned no rationale)"

#: Shown in error messages: a planner failure is fatal for the run, so the shape must be obvious.
PLAN_SHAPE_HINT = (
    '{"tasks": [{"id": "1", "description": "...", "files": [], "dependencies": []}], '
    '"rationale": "..."}'
)


def _task_fields(raw: Any) -> dict[str, Any] | None:
    """Map one raw plan entry onto ``TaskModel`` kwargs, or ``None`` if it is unusable.

    ``id`` is handled separately (it can be missing). Anything the schema types strictly is
    coerced here so a single odd field cannot invalidate the whole plan.
    """
    if isinstance(raw, str):
        return {"description": raw.strip()} if raw.strip() else None
    if not isinstance(raw, dict):
        return None

    fields: dict[str, Any] = {}
    for key in DESCRIPTION_KEYS:
        description = as_text(raw.get(key))
        if description:
            fields["description"] = description
            break
    else:
        # No prose field: keep the task and surface whatever the model did send, so a nearly
        # correct plan still makes progress instead of being dropped.
        rendered = ", ".join(f"{key}={raw[key]!r}" for key in sorted(raw))
        fields["description"] = truncate(
            rendered or "task described only by its id", MAX_SYNTHETIC_DESCRIPTION_CHARS
        )

    files = as_str_list(raw.get("files"))
    if files:
        fields["files"] = files
    for key in DEPENDENCY_KEYS:
        dependencies = as_str_list(raw.get(key))
        if dependencies:
            fields["dependencies"] = dependencies
            break
    return fields


def _unique_id(task_id: str, taken: set[str]) -> str:
    """Return ``task_id``, or the first free ``task_id-N`` variant (the orchestrator's rule)."""
    index = 2
    while f"{task_id}-{index}" in taken:
        index += 1
    return f"{task_id}-{index}"


def _plan_error(reason: str, response: str) -> str:
    """Build the actionable message raised when no usable plan can be recovered."""
    return (
        f"Planner returned no usable plan: {reason}. Expected JSON shaped like "
        f"{PLAN_SHAPE_HINT}. Model response: "
        f"{truncate(response, MAX_ERROR_PREVIEW_CHARS) or '<empty>'}"
    )


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        plan = self._to_plan(response)
        logger.info(
            "Planner produced %d task(s): %s",
            len(plan.tasks),
            ", ".join(task.id for task in plan.tasks),
        )
        return plan

    @staticmethod
    def _to_plan(response: str) -> Plan:
        """Parse, validate and normalise a model response into a :class:`Plan`.

        ``extract_json`` absorbs fences, surrounding prose, trailing commas and python-style
        literals. What survives is then reshaped defensively: positional ids for tasks without
        one, de-duplicated ids, coerced list fields. :class:`ValueError` is raised only when
        nothing usable is left -- the orchestrator halts on that, so the message says why.
        """
        try:
            data = extract_json(response)
        except (ValueError, TypeError) as exc:
            raise ValueError(_plan_error(str(exc), response)) from exc

        if not isinstance(data, dict):
            raise ValueError(
                _plan_error(f"top-level JSON is {type(data).__name__}, not an object", response)
            )

        raw_tasks = data.get("tasks")
        if isinstance(raw_tasks, dict):
            # `{"tasks": {"1": {...}}}` keyed-by-id plans show up often enough to accept.
            expanded: list[Any] = []
            for key, value in raw_tasks.items():
                if isinstance(value, dict):
                    expanded.append({"id": key, **value})
                else:
                    expanded.append({"id": key, "description": value})
            raw_tasks = expanded
        if not isinstance(raw_tasks, list):
            raise ValueError(_plan_error('"tasks" is missing or is not a list', response))

        tasks: list[TaskModel] = []
        taken: set[str] = set()
        for index, raw in enumerate(raw_tasks):
            fields = _task_fields(raw)
            if fields is None:
                logger.warning("Planner task #%d is unusable (%r); skipping it.", index + 1, raw)
                continue
            task_id = as_text(raw.get("id")) if isinstance(raw, dict) else ""
            if not task_id:
                task_id = str(index + 1)
            elif task_id in taken:
                unique = _unique_id(task_id, taken)
                logger.warning("Planner reused task id %r; registered as %r.", task_id, unique)
                task_id = unique
            taken.add(task_id)
            tasks.append(TaskModel(id=task_id, **fields))

        if not tasks:
            raise ValueError(_plan_error("the plan contains no usable tasks", response))

        rationale = as_text(data.get("rationale"))
        return Plan(tasks=tasks, rationale=rationale or DEFAULT_RATIONALE)
