from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.core.json_utils import extract_json
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Long enough to debug a bad response, short enough for a log line / next-cycle directive.
_SNIPPET_LIMIT = 500


def _snippet(text: str, limit: int = _SNIPPET_LIMIT) -> str:
    """Return a truncated copy of ``text`` for inclusion in error messages."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _coerce_task_list(value: Any) -> list[Any]:
    """Return ``value`` as a list.

    A model that answers ``{"tasks": {...}}`` instead of ``{"tasks": [{...}]}``
    would otherwise fail pydantic validation, so a lone task object is wrapped
    into a single-element list. ``None`` becomes an empty list.
    """
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _task_id(entry: Any) -> str | None:
    """Return a usable id for a raw plan entry, or ``None`` when it has none.

    Non-dict entries (a bare string, say) and entries with a missing or empty id
    yield ``None`` so the caller can drop them. Numeric ids are stringified,
    which also makes ``1`` and ``"1"`` collide as duplicates, as they should.
    """
    if not isinstance(entry, dict):
        return None
    raw = entry.get("id")
    if raw is None:
        return None
    ident = str(raw).strip()
    return ident or None


def _dedupe_tasks(entries: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """Drop entries with a missing or duplicated id, keeping the first occurrence.

    Returns the surviving entries plus the number of dropped ones. Duplicate ids
    break the orchestrator's task correlation and status reporting, so they never
    reach the plan.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for entry in entries:
        ident = _task_id(entry)
        if ident is None or ident in seen:
            dropped += 1
            continue
        seen.add(ident)
        kept.append({**entry, "id": ident})
    return kept, dropped


def build_plan(data: Any, max_parallel_tasks: int) -> Plan:
    """Validate and normalise a parsed LLM plan payload into a :class:`Plan`.

    Raises ``TypeError`` if the payload is not an object, or ``ValueError``
    (a pydantic ``ValidationError`` is one) if it cannot be coerced into a plan;
    the caller adds the raw response for context.
    """
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object with a 'tasks' key, got {type(data).__name__}")

    tasks, dropped = _dedupe_tasks(_coerce_task_list(data.get("tasks")))

    notes: list[str] = []
    if dropped:
        notes.append(f"Dropped {dropped} planned task(s) with a missing or duplicated id.")

    # The orchestrator also caps concurrency; capping here keeps the plan honest.
    cap = max(0, max_parallel_tasks)
    if len(tasks) > cap:
        omitted = len(tasks) - cap
        notes.append(
            f"Truncated the plan to the first {cap} task(s) to respect "
            f"max_parallel_tasks={cap}; {omitted} further task(s) deferred."
        )
        tasks = tasks[:cap]

    rationale = data.get("rationale")
    rationale = rationale.strip() if isinstance(rationale, str) else ""
    if notes:
        rationale = f"{rationale} {' '.join(notes)}".strip()

    # Validate each entry explicitly rather than leaning on Plan's coercion of
    # raw dicts; a bad entry still raises a pydantic ValidationError, which is a
    # ValueError, so the caller's except clause is unchanged.
    return Plan(tasks=[TaskModel.model_validate(t) for t in tasks], rationale=rationale)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            # Tolerates ```json fences, leading prose and trailing commentary.
            return build_plan(extract_json(response), self.client.settings.max_parallel_tasks)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse plan from LLM: {e}\nResponse: {_snippet(response)}"
            ) from e
