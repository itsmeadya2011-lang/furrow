from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient, LLMResponseError

if TYPE_CHECKING:
    from furrow.config import Settings

#: Documented ceiling on the number of tasks a plan may contain.
MAX_TASKS: Final[int] = 5
#: Characters of an offending response kept in an error message.
SNIPPET_LIMIT: Final[int] = 300
#: Appended to ``rationale`` when a plan is longer than ``MAX_TASKS``.
TRUNCATION_NOTE: Final[str] = (
    "Plan truncated by furrow to the {limit}-task ceiling; the remaining tasks were dropped."
)


def _snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Return a single-line, length-capped excerpt of ``text`` for error messages."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _normalize_plan(plan: Plan, max_tasks: int = MAX_TASKS) -> Plan:
    """Return a copy of ``plan`` that is safe for the orchestrator's scheduler.

    Tasks whose ``description`` is blank are dropped, blank or duplicated ids are
    re-assigned the next free sequential number, and ``dependencies`` are remapped
    onto the new ids (a duplicated original id resolves to the first task that
    claimed it). Dependencies pointing at ids that are no longer present -- because
    the task was dropped or truncated away -- are removed so the wave scheduler is
    not asked to wait for a task that does not exist. At most ``max_tasks`` tasks are
    kept; when the plan is longer a note is appended to ``rationale``.
    """
    seen: set[str] = set()
    id_map: dict[str, str] = {}
    next_id = 1
    tasks: list[TaskModel] = []
    for task in plan.tasks:
        description = task.description.strip()
        if not description:
            continue
        original = task.id.strip()
        task_id = original
        if not task_id or task_id in seen:
            while str(next_id) in seen:
                next_id += 1
            task_id = str(next_id)
            seen.add(task_id)
            next_id += 1
        else:
            seen.add(task_id)
        if original:
            id_map.setdefault(original, task_id)
        tasks.append(task.model_copy(update={"id": task_id, "description": description}))

    limit = max(0, max_tasks)
    truncated = len(tasks) > limit
    tasks = tasks[:limit]

    surviving = {task.id for task in tasks}
    resolved: list[TaskModel] = []
    for task in tasks:
        dependencies: list[str] = []
        for dependency in task.dependencies:
            target = id_map.get(dependency.strip())
            if target is None or target not in surviving or target in dependencies:
                continue
            dependencies.append(target)
        resolved.append(task.model_copy(update={"dependencies": dependencies}))

    rationale = plan.rationale.strip()
    if truncated:
        note = TRUNCATION_NOTE.format(limit=limit)
        rationale = f"{rationale} {note}".strip()
    return Plan(tasks=resolved, rationale=rationale)


def _build_prompt(goal: str, context: str | None = None) -> str:
    """Assemble the planner prompt, folding in previous-cycle context when present."""
    sections = [PLANNER_PROMPT, f"Goal: {goal}"]
    if context and context.strip():
        sections.append(f"Context:\n{context.strip()}")
    return "\n\n".join(sections)


def _build_repair_prompt(prompt: str, error: str) -> str:
    """Ask the model to re-emit the plan as bare JSON after a parse failure.

    ``error`` is the ``LLMResponseError`` message, which embeds an excerpt of the
    offending response.
    """
    return (
        f"{prompt}\n\n"
        f"Your previous reply could not be parsed as JSON: {error}\n\n"
        "Reply again with the same JSON object. Return raw JSON only. "
        "Do not wrap it in markdown code fences and do not add any prose "
        "before or after the JSON object."
    )


class PlannerAgent:
    """Turns a goal into a normalised plan of at most ``MAX_TASKS`` tasks."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, context: str | None = None) -> Plan:
        """Return a plan for ``goal``, repairing one unparseable response.

        The first response is parsed with ``complete_json``. If it is not JSON, one
        repair request is made; a second failure raises ``LLMResponseError`` quoting
        both responses.
        """
        prompt = _build_prompt(goal, context)
        model = self.client.settings.planner_model
        try:
            data = await self.client.complete_json(prompt, model=model)
        except LLMResponseError as first:
            try:
                data = await self.client.complete_json(
                    _build_repair_prompt(prompt, str(first)), model=model
                )
            except LLMResponseError as second:
                raise LLMResponseError(
                    "Planner returned unparseable JSON twice; first attempt: "
                    f"{_snippet(str(first))}; repair attempt: {_snippet(str(second))}"
                ) from second
        return _normalize_plan(self._to_plan(data))

    @staticmethod
    def _to_plan(data: dict[str, Any]) -> Plan:
        """Validate the model payload, tolerating a missing rationale."""
        if not isinstance(data, Mapping):
            raise LLMResponseError(
                "Planner response was not a JSON object; got "
                f"{type(data).__name__}: {_snippet(repr(data))}"
            )
        payload = dict(data)
        payload.setdefault("rationale", "")
        return Plan(**payload)