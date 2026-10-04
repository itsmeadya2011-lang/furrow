"""Goal decomposition for Furrow.

:class:`PlannerAgent` turns a free-form goal into a validated
:class:`furrow.config.Plan` that ``furrow.core.orchestrator.Orchestrator`` can execute.
The model's JSON is treated as UNTRUSTED input: ``LLMClient.complete_json`` guarantees the
reply parses, not that the parsed value is a plan, so every field is coerced before it
reaches pydantic. One sloppy task object must never take down a whole cycle.

Two deliberate decisions live here:

* **No task-count cap.** ``settings.max_parallel_tasks`` is deliberately NOT used to
  truncate the plan. Truncating silently drops planned work, and the orchestrator already
  enforces the cap with an ``asyncio.Semaphore`` (``Orchestrator._execute``), so keeping
  every task is safe. Only ids are repaired, because duplicate ids would collapse two
  tasks into one inside ``Orchestrator._merge_plan``.
* **Unknown keys are dropped explicitly.** The contract below builds ``TaskModel`` from
  the four keys the model is allowed to send rather than splatting the raw mapping, so a
  hallucinated key can never reach the model layer.

Constructor shape is a contract with the orchestrator:
``PlannerAgent(client=client, settings=settings)``. The optional ``bus=`` only adds
``PLAN`` and ``ERROR`` events to the live feed; it changes no return value.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.events import EventType
from furrow.llm import LLMClient, LLMResponseError

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.events import EventBus

MAX_DESCRIPTION_CHARS = 400
MAX_RATIONALE_CHARS = 2_000
EVENT_DESCRIPTION_CHARS = 120
EVENT_RATIONALE_CHARS = 500
EVENT_TASK_LIMIT = 20


def _head(text: str, limit: int) -> str:
    """Return at most the first `limit` characters of `text`, marking any truncation."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars truncated]"


def _string_list(value: Any) -> list[str]:
    """Coerce `value` to a list of non-empty strings, dropping every other entry."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _raw_task(entry: Any, index: int) -> dict[str, Any]:
    """Coerce one untrusted task entry into the four keys `TaskModel` accepts.

    `id` is synthesised from the list position when absent or blank, and `description`
    is truncated / stringified rather than rejected, so a partially broken task degrades
    into a weak task instead of destroying the whole plan.
    """
    data: dict[str, Any] = entry if isinstance(entry, dict) else {"description": entry}
    identifier = data.get("id")
    if isinstance(identifier, (int, float)) and not isinstance(identifier, bool):
        # POLISH (wave 5b): the old comment here claimed a numeric `id` kept its value as
        # a string "so a [1] dependency entry still resolves to this task". `_string_list`
        # keeps only real `str` entries, so numeric dependencies are DROPPED, not
        # coerced - id coercion alone cannot match them. Documented, intentional behaviour.
        identifier = str(identifier)
    if not isinstance(identifier, str) or not identifier.strip():
        identifier = str(index + 1)
    description = data.get("description")
    if description is None:
        description = f"Task {identifier.strip()}: the planner returned no description."
    elif not isinstance(description, str):
        description = str(description)
    return {
        "id": identifier.strip(),
        "description": _head(description.strip(), MAX_DESCRIPTION_CHARS),
        "files": _string_list(data.get("files")),
        "dependencies": _string_list(data.get("dependencies")),
    }


def _unique_ids(tasks: list[dict[str, Any]]) -> dict[str, str]:
    """Give every task a unique id, returning the old-id -> new-id mapping.

    Colliding ids are renamed with a numeric suffix rather than dropped, and the mapping
    lets dependencies follow the rename. When one old id maps to several new ones the
    first mapping wins: a dependency pointing at an ambiguous id is already broken.
    """
    used: set[str] = set()
    remap: dict[str, str] = {}
    for task in tasks:
        original = task["id"]
        candidate = original
        if candidate in used:
            suffix = 2
            while f"{candidate}-{suffix}" in used:
                suffix += 1
            candidate = f"{candidate}-{suffix}"
        used.add(candidate)
        task["id"] = candidate
        remap.setdefault(original, candidate)
    return remap


def _rewrite_dependencies(tasks: list[dict[str, Any]], remap: dict[str, str]) -> None:
    """Follow renamed ids and drop self-references and duplicates from dependencies."""
    for task in tasks:
        rewritten: list[str] = []
        for dependency in task["dependencies"]:
            target = remap.get(dependency, dependency)
            if target == task["id"] or target in rewritten:
                continue
            rewritten.append(target)
        task["dependencies"] = rewritten


def _plan_from_payload(payload: Any) -> Plan:
    """Validate and normalise the model's JSON payload into a `Plan`.

    Raises `LLMResponseError` when the payload is not an object or carries no `tasks`
    array - the two shapes no amount of defensiveness can repair.
    """
    if not isinstance(payload, dict):
        raise LLMResponseError(
            f"Planner reply must be a JSON object, got {type(payload).__name__}"
        )
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise LLMResponseError("Planner reply is missing a 'tasks' array")
    tasks = [_raw_task(entry, index) for index, entry in enumerate(raw_tasks)]
    _rewrite_dependencies(tasks, _unique_ids(tasks))
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return Plan(
        tasks=[TaskModel(**task) for task in tasks],
        rationale=_head(rationale.strip(), MAX_RATIONALE_CHARS),
    )


class PlannerAgent:
    """Turns a goal into an executable, validated plan."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.client: LLMClient = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings
        self.bus = bus

    async def plan(self, goal: str) -> Plan:
        """Decompose `goal` into a plan.

        Raises `LLMResponseError` when the reply parses but is not a plan. Every other
        failure (auth, rate limit, transport) propagates unchanged and is NOT converted
        into an empty plan: ``Orchestrator._cycle`` already isolates planner exceptions
        into an ERROR event and a stalled cycle, and that contract is preserved here.
        """
        prompt = f"Goal:\n{goal}\n\nProduce the task breakdown for this goal."
        try:
            payload = await self.client.complete_json(
                prompt,
                system=PLANNER_PROMPT,
                model=self.client.settings.planner_model,
            )
            plan = _plan_from_payload(payload)
        except json.JSONDecodeError as err:
            # `complete_json` already converts parse failures into LLMResponseError, but a
            # caller-supplied client that leaks the raw decoder error must not turn this
            # method's documented failure type into something else.
            self._emit_error("Planner reply was not valid JSON", err)
            raise LLMResponseError(f"Planner reply was not valid JSON: {err}") from err
        except Exception as err:
            self._emit_error("Planner failed", err)
            raise
        self._emit_plan(plan)
        return plan

    def _emit(self, event: EventType, message: str, **data: Any) -> None:
        """Emit on the bus when one was supplied; a missing bus is not an error."""
        if self.bus is not None:
            self.bus.emit(event, message, **data)

    def _emit_error(self, context: str, err: BaseException) -> None:
        self._emit(
            EventType.ERROR,
            f"{context}: {err}",
            context=context,
            error=_head(str(err), EVENT_RATIONALE_CHARS),
        )

    def _emit_plan(self, plan: Plan) -> None:
        self._emit(
            EventType.PLAN,
            f"Planned {len(plan.tasks)} task(s)",
            rationale=_head(plan.rationale, EVENT_RATIONALE_CHARS),
            task_count=len(plan.tasks),
            tasks=[
                {
                    "id": task.id,
                    "description": _head(task.description, EVENT_DESCRIPTION_CHARS),
                    "files": len(task.files),
                    "dependencies": len(task.dependencies),
                }
                for task in plan.tasks[:EVENT_TASK_LIMIT]
            ],
        )
